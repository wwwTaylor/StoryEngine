from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import json
import re

ROOT=Path(__file__).resolve().parents[1]
errors=[]
class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=set();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:
            if a['id'] in self.ids: errors.append('Duplicate id: '+a['id'])
            self.ids.add(a['id'])
        for attr in ['src','href','poster','data-zoom']:
            if attr in a:self.links.append(a[attr])
page=Page();page.feed((ROOT/'index.html').read_text(encoding='utf-8'))
for link in page.links:
    parsed=urlparse(link)
    if parsed.scheme or parsed.netloc:continue
    if parsed.path:
        target=(ROOT/unquote(parsed.path)).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():errors.append('Missing local asset: '+link)
    elif parsed.fragment and parsed.fragment not in page.ids:errors.append('Unknown anchor: '+link)
data=json.loads((ROOT/'assets/site-data.json').read_text(encoding='utf-8'))
assert len(data['showcases'])==5 and len(data['comparisons'])==3
media=[]
for item in data['showcases']:
    media.append(item['video']);media.append(item['poster'])
for comparison in data['comparisons']:
    for item in comparison['methods'].values():media.extend([item['video'],item['poster']])
for relative in media:
    if not (ROOT/relative).is_file():errors.append('Missing data asset: '+relative)
metrics=['ECS','APR','SPS','LAR','GPC','MIR','LCS']
for row in data['mainResults']:
    if abs(sum(row[m] for m in metrics)/7-row['Avg'])>0.00011:errors.append('Avg mismatch: '+str(row))
for path in ROOT.rglob('*'):
    if not path.is_file() or '.git' in path.parts:continue
    if path.stat().st_size>=100_000_000:errors.append('File too large for normal GitHub upload: '+str(path))
assert not errors,'\n'.join(errors)
print('PASS: unique HTML IDs, local links and anchors, 14 video sources, paper metrics, and file-size limits.')
