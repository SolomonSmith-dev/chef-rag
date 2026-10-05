# Corpus sources

Public-domain only. `scripts/fetch_corpus.py` fills in `data/raw/manifest.json`
(retrieval date and sha256 per file). The retrieval dates below stay "not yet
retrieved" until you run `docs/corpus-runbook.md` on a machine with network access.
The build sandbox blocked every host below, so none of these URLs or Gutenberg ids
were opened by the author of this table. The script's `expect` phrase check
fails loudly if an id or page is wrong.

| name | title | URL | license basis | retrieved |
|---|---|---|---|---|
| escoffier_guide_to_modern_cookery | A Guide to Modern Cookery (Escoffier, 1907) | https://www.gutenberg.org/cache/epub/47703/pg47703.txt | Public domain in the US (published 1907). Gutenberg header and license text stripped. | not yet retrieved |
| farmer_boston_cooking_school_cook_book | The Boston Cooking-School Cook Book (Farmer, 1896) | https://www.gutenberg.org/cache/epub/65/pg65.txt | Public domain in the US (published 1896). | not yet retrieved |
| beeton_household_management | Mrs Beeton's Book of Household Management (1861) | https://www.gutenberg.org/cache/epub/10136/pg10136.txt | Public domain in the US (published 1861). Kitchen-management text. | not yet retrieved |
| fda_food_code_2022 | FDA Food Code 2022 | https://www.fda.gov/media/164194/download | US government work, no copyright (17 U.S.C. 105). | not yet retrieved |
| fsis_safe_minimum_internal_temperature_chart | USDA FSIS Safe Minimum Internal Temperature Chart | https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart | US government work (17 U.S.C. 105). | not yet retrieved |
| fsis_danger_zone | USDA FSIS Danger Zone (40 F - 140 F) | https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/danger-zone-40f-140f | US government work (17 U.S.C. 105). | not yet retrieved |

Technique-focused text: Escoffier and Farmer. Kitchen-management text: Beeton.
Because Beeton (1861) and Escoffier (1907) predate modern food-safety science, the
eval set tests that answers prefer the FDA/FSIS sources on safety questions.

## Test fixtures

`tests/fixtures/corpus/` holds three tiny synthetic documents written for tests.
They are not part of the corpus and carry no real content claims.

## Original documents

`data/raw/original/` is reserved for documents you write yourself
(`source_type: original`). Nothing there is generated.
