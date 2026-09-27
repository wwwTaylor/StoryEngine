# StoryEngine project website

Static project website for **StoryEngine: A State-Grounded Agentic Framework for Video Storytelling**.

Website: https://wwwtaylor.github.io/StoryEngine/

Code: https://github.com/wwwTaylor/StoryEngine

This website is published from the `gh-pages` branch. The repository's `main` branch contains the StoryEngine implementation.

The page includes the project title and authors, Code and demo links, abstract, five one-minute demonstrations, three comparison groups, four research figures, benchmark results, and BibTeX with a copy button. The local manuscript PDF is excluded from publication; a paper link can be added when the arXiv URL is available.

## Local checks

```bash
python tools/check_site.py
node --check app.js
node --check site-data.js
```

Open `index.html` for a basic preview. For reliable video seeking, use a local HTTP server supporting byte-range requests.

## Publishing

GitHub Pages serves the root directory of `gh-pages`. Commit reviewed website changes on this branch and push to update the site. The `.nojekyll` file enables direct static-file publishing.

The tracked assets include all 14 demo/comparison videos, their posters, four WebP figures, and the social preview image. Local PDFs, unused preview media, credentials, and the unused Actions deployment template are excluded by `.gitignore`.

Layout and interaction references: [SCOPE](https://z2tong.github.io/SCOPE/), [WorldMind](https://teawhite.cn/WorldMind/), and [StatePlay](https://jimntu.github.io/stateplay_page/). Reference-site media were not imported.
