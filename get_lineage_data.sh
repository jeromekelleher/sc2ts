#!/bin/bash
# Shell script to download pango lineage data from cov-lineages.

wget https://github.com/cov-lineages/lineages-website/raw/refs/heads/master/_data/lineage_data.full.json
# Cut down the fields to remove country counts etc.
cat lineage_data.full.json | jq '[.[]|{Lineage,"Earliest date","Latest date",Description}]' > sc2ts/data/lineages.json
