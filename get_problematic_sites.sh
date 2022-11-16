#!/bin/bash
# Shell script to download the list of problematic_sites from the UCSC list.
# Note: the resulting text file should also be committed to git.

rm -f problematic_sites_sarsCov2.vcf
wget https://raw.githubusercontent.com/W-L/ProblematicSites_SARS-CoV2/master/problematic_sites_sarsCov2.vcf
bcftools view -H problematic_sites_sarsCov2.vcf | cut -f2 > sarscov2ts/data/problematic_sites.txt
