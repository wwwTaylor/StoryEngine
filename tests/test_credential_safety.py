from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

import httpx
import yaml
from pydantic import ValidationError

from story_engine.bootstrap import run_command
from story_engine.cli import main
from story_engine.config import AppConfig, ProviderSpec, SecretValue
from story_engine.domain.request import ProvidedAsset
from story_engine.errors import ConfigurationError, ProviderError
from story_engine.providers.transport import HttpTransport
from story_engine.run_spec import load_run_spec
from story_engine.run_state import RunStore
from story_engine.security import redact_data, redact_text

ROOT = Path(__file__).resolve().parents[1]
# Synthetic test data; never loaded from the environment or sent to a real server.
DUMMY = "test-only-credential-value-0123456789"


class CredentialConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.definition = yaml.safe_load((ROOT / "examples/tiny_request.yaml").read_text("utf-8"))

    def write_definition(self, value: dict) -> Path:
        path = self.root / "request.local.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    def test_rejects_credential_spellings_and_nested_values(self) -> None:
        names = (
            "password", "passwd", "pwd", "passphrase", "access_key", "AccessKeyId",
            "secretAccessKey", "refresh_token", "apiToken", "API-KEY", "clientSecret",
            "X-Goog-Api-Key", "Authorization", "Proxy-Authorization", "Cookie", "privateKey",
        )
        for name in names:
            with self.subTest(name=name):
                with self.assertRaises(ValidationError) as raised:
                    ProviderSpec(
                        adapter="openai_responses", model="example",
                        options={"nested": [{name: DUMMY}]},
                    )
                self.assertNotIn(DUMMY, str(raised.exception))

    def test_encoded_option_pairs_cannot_bypass_credential_filter(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            ProviderSpec(
                adapter="openai_responses", model="example",
                options=(("nested", json.dumps({"accessKey": DUMMY})),),
            )
        self.assertNotIn(DUMMY, str(raised.exception))

    def test_example_validates_without_environment_keys_and_round_trips(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            spec = load_run_spec(self.write_definition(self.definition))
        saved = spec.config.redacted_dict()
        self.assertEqual(AppConfig.model_validate(saved), spec.config)
        self.assertEqual(saved["providers"]["planner"]["api_key_env"], "STORY_ENGINE_GATEWAY_KEY")
        self.assertEqual(saved["providers"]["judge"]["options"]["credential_mode"], "bearer")

    def test_persistence_redacts_even_when_model_validation_is_bypassed(self) -> None:
        config = AppConfig.model_validate(self.definition["runtime"])
        unsafe = config.providers.planner.model_copy(update={
            "options": (("nested", json.dumps([{"password": DUMMY}])),),
        })
        config = config.model_copy(update={
            "providers": config.providers.model_copy(update={"planner": unsafe}),
        })
        saved = config.redacted_dict()
        self.assertNotIn(DUMMY, json.dumps(saved))
        self.assertEqual(saved["providers"]["planner"]["options"]["nested"][0]["password"], "***")

    def test_unknown_options_fail_before_run_directory_creation(self) -> None:
        self.definition["runtime"]["providers"]["planner"]["options"]["opaque_value"] = DUMMY
        args = SimpleNamespace(command="run", definition=self.write_definition(self.definition), run_id="test")
        with patch("story_engine.bootstrap.RunStore") as store:
            with self.assertRaises(ConfigurationError) as raised:
                run_command(args)
        store.assert_not_called()
        self.assertNotIn(DUMMY, str(raised.exception))

    def test_cli_hides_secret_values_and_unknown_field_names(self) -> None:
        for changes in (
            {"password": DUMMY},
            {"max_attachments": DUMMY},
            {DUMMY: "invalid-extra-field"},
        ):
            with self.subTest(changes=list(changes)):
                value = copy.deepcopy(self.definition)
                value["runtime"]["providers"]["planner"]["options"].update(changes)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    status = main(["validate", str(self.write_definition(value))])
                self.assertEqual(status, 2)
                self.assertNotIn(DUMMY, stderr.getvalue())
                self.assertNotIn("input_value=", stderr.getvalue())

    def test_cli_does_not_echo_invalid_yaml_source_line(self) -> None:
        path = self.root / "broken.local.yaml"
        path.write_text("runtime: [" + DUMMY, encoding="utf-8")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["validate", str(path)])
        self.assertEqual(status, 2)
        self.assertIn("line", stderr.getvalue())
        self.assertNotIn(DUMMY, stderr.getvalue())

    def test_asset_urls_reject_query_fragment_and_user_info(self) -> None:
        for url in (
            "https://assets.example/image.png?token=" + DUMMY,
            "https://assets.example/image.png?X-Amz-Signature=" + DUMMY,
            "https://assets.example/image.png#" + DUMMY,
            "https://user:" + DUMMY + "@assets.example/image.png",
        ):
            with self.subTest(url=url.split("?")[0]):
                with self.assertRaises(ValidationError) as raised:
                    ProvidedAsset(asset_id="image", kind="image", description="example", uri=url)
                self.assertNotIn(DUMMY, str(raised.exception))

    def test_invalid_asset_url_does_not_reach_run_storage_or_cli_logs(self) -> None:
        self.definition["provided_assets"] = [{
            "asset_id": "image", "kind": "image", "description": "example",
            "uri": "https://assets.example/image.png?token=" + DUMMY,
        }]
        stderr = io.StringIO()
        with patch("story_engine.bootstrap.RunStore") as store, redirect_stderr(stderr):
            status = main(["run", str(self.write_definition(self.definition))])
        self.assertEqual(status, 2)
        self.assertNotIn(DUMMY, stderr.getvalue())
        store.assert_not_called()

    def test_public_asset_url_and_local_path_remain_supported(self) -> None:
        for source in ({"uri": "https://assets.example/image.png"}, {"path": Path("image.png")}):
            asset = ProvidedAsset(asset_id="image", kind="image", description="example", **source)
            self.assertTrue(asset.path is not None or asset.uri is not None)

    def test_custom_run_directories_ignore_generated_contents(self) -> None:
        store = RunStore(self.root / "custom-output" / "example")
        self.assertEqual((store.run_dir / ".gitignore").read_text("utf-8"), "*\n")

    def test_recursive_export_and_text_redaction_hide_credentials(self) -> None:
        source = {"nested": [{"apiToken": DUMMY}], "url": "https://assets.example/a?sig=" + DUMMY}
        safe = redact_data(source)
        self.assertNotIn(DUMMY, json.dumps(safe))
        self.assertEqual(source["nested"][0]["apiToken"], DUMMY)
        self.assertEqual(redact_data({"max_tokens": 100}), {"max_tokens": 100})

    def test_secret_repr_and_encoded_log_values_do_not_reveal_key(self) -> None:
        value = DUMMY + "/+="
        self.assertNotIn(DUMMY, repr(SecretValue(value)))
        self.assertNotIn(DUMMY, str(SecretValue(value)))
        result = redact_text("failure: " + quote(value, safe=""), secrets=(value,))
        self.assertNotIn(DUMMY, result)


class CredentialTransportTests(unittest.IsolatedAsyncioTestCase):
    async def transport(
        self, handler, *, retries: int = 0, headers=None, auth=None, params=None
    ) -> HttpTransport:
        client = await self.enterAsyncContext(httpx.AsyncClient(
            base_url="https://provider.example/v1/",
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
            headers=headers,
            auth=auth,
            params=params,
        ))
        return HttpTransport(
            base_url="https://provider.example/v1", credential=SecretValue(DUMMY),
            timeout_seconds=1, max_retries=retries, client=client,
            credential_header="x-goog-api-key", credential_prefix="", sleeper=AsyncMock(),
        )

    async def test_cross_origin_media_redirect_strips_all_authentication(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(302, headers={"Location": "https://cdn.example/image?sig=download-ticket"})
            return httpx.Response(200, content=b"image")

        transport = await self.transport(handler, headers={
            "Authorization": "Bearer " + DUMMY, "Cookie": "session=" + DUMMY,
            "X-Secondary-Api-Key": DUMMY,
        }, auth=httpx.BasicAuth("example", DUMMY))
        result = await transport.request("GET", "images/123")
        self.assertEqual(result.response.content, b"image")
        self.assertEqual(seen[0].headers["x-goog-api-key"], DUMMY)
        self.assertEqual(seen[1].url.host, "cdn.example")
        self.assertNotIn(DUMMY, str(seen[1].headers))
        for name in ("Authorization", "Cookie", "x-goog-api-key", "X-Secondary-Api-Key"):
            self.assertNotIn(name, seen[1].headers)

    async def test_same_origin_redirect_retains_required_authentication(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(307, headers={"Location": "https://provider.example:443/v1/new"})
            return httpx.Response(200, json={"ok": True})

        transport = await self.transport(handler)
        await transport.request("POST", "old", json_body={"prompt": "example"})
        self.assertEqual(len(seen), 2)
        for request in seen:
            self.assertEqual(request.headers["x-goog-api-key"], DUMMY)
            self.assertEqual(request.method, "POST")
            self.assertEqual(json.loads(request.content), {"prompt": "example"})

    async def test_external_download_does_not_inherit_client_query_credentials(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200)

        transport = await self.transport(handler, params={"api_key": DUMMY})
        await transport.request("GET", "https://cdn.example/image?sig=download-ticket")
        self.assertEqual(dict(seen[0].url.params), {"sig": "download-ticket"})
        self.assertNotIn(DUMMY, str(seen[0].url))

    async def test_cross_origin_redirect_never_forwards_prompt_or_multipart_body(self) -> None:
        for body in ({"json_body": {"prompt": "private-story"}}, {"files": [("image", ("a.png", b"private-image", "image/png"))]}):
            with self.subTest(body=list(body)):
                seen = []

                def handler(request):
                    seen.append(request)
                    return httpx.Response(307, headers={"Location": "https://cdn.example/collect"})

                transport = await self.transport(handler)
                with self.assertRaises(ProviderError):
                    await transport.request("POST", "generate", **body)
                self.assertEqual(len(seen), 1)

    async def test_network_relative_external_url_has_no_credentials(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200)

        transport = await self.transport(handler)
        await transport.request("GET", "//cdn.example/image")
        self.assertEqual(seen[0].url.host, "cdn.example")
        self.assertNotIn("x-goog-api-key", seen[0].headers)

    async def test_returning_to_provider_does_not_restore_credentials(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            locations = ["https://cdn.example/a", "https://provider.example/v1/other"]
            if len(seen) <= 2:
                return httpx.Response(302, headers={"Location": locations[len(seen) - 1]})
            return httpx.Response(200)

        transport = await self.transport(handler)
        await transport.request("GET", "media")
        self.assertEqual(len(seen), 3)
        self.assertNotIn("x-goog-api-key", seen[2].headers)

    async def test_https_downgrade_is_not_followed(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(302, headers={"Location": "http://provider.example/v1/media"})

        transport = await self.transport(handler)
        with self.assertRaises(ProviderError):
            await transport.request("GET", "media")
        self.assertEqual(len(seen), 1)

    async def test_redirect_limit_is_enforced(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(302, headers={"Location": "/v1/loop"})

        transport = await self.transport(handler)
        with self.assertRaises(ProviderError):
            await transport.request("GET", "loop")
        self.assertEqual(len(seen), 11)

    async def test_errors_redact_before_truncating(self) -> None:
        transport = await self.transport(lambda request: httpx.Response(500, text="x" * 980 + DUMMY))
        with self.assertRaises(ProviderError) as raised:
            await transport.request("GET", "models")
        self.assertNotIn(DUMMY[:20], str(raised.exception))
        self.assertIn("***", str(raised.exception))

    async def test_error_urls_and_other_credential_fields_are_redacted(self) -> None:
        other = "another-test-only-secret"
        body = json.dumps({"password": other, "url": "https://cdn.example/a?sig=" + other})
        transport = await self.transport(lambda request: httpx.Response(400, text=body))
        with self.assertRaises(ProviderError) as raised:
            await transport.request("GET", "models")
        self.assertNotIn(other, str(raised.exception))

    async def test_timeout_errors_redact_known_key(self) -> None:
        def handler(request):
            raise httpx.ReadTimeout("test transport error " + DUMMY, request=request)

        transport = await self.transport(handler)
        with self.assertRaises(ProviderError) as raised:
            await transport.request("GET", "models")
        self.assertNotIn(DUMMY, str(raised.exception))

    async def test_existing_rate_limit_retry_still_succeeds(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(429 if len(seen) == 1 else 200)

        transport = await self.transport(handler, retries=1)
        result = await transport.request("GET", "models")
        self.assertEqual(result.retries, 1)
        self.assertEqual(len(seen), 2)

    async def test_url_with_user_info_is_rejected_without_echoing_password(self) -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200)

        transport = await self.transport(handler)
        with self.assertRaises(ProviderError) as raised:
            await transport.request("GET", "https://user:" + DUMMY + "@cdn.example/a")
        self.assertNotIn(DUMMY, str(raised.exception))
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
