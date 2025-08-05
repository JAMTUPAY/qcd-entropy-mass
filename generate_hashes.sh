#!/bin/bash
# Generate SHA256 hashes for QCD Entropy-Mass Discovery

echo "Generating SHA256 hashes for QCD Entropy-Mass Discovery"
echo "========================================================"
echo ""

# Create hash file
HASHFILE="SHA256SUMS.txt"
echo "# SHA256 Checksums for QCD Entropy-Mass Discovery" > $HASHFILE
echo "# Generated on $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> $HASHFILE
echo "# DOI: 10.5281/zenodo.16743904" >> $HASHFILE
echo "" >> $HASHFILE

# Hash the paper files
echo "Paper files:" >> $HASHFILE
cd paper
sha256sum Universal_Entropy_Mass_Relation_in_QCD.pdf >> ../$HASHFILE
sha256sum Universal_Entropy_Mass_Relation_in_QCD.tex >> ../$HASHFILE
sha256sum main.tex >> ../$HASHFILE
cd ..
echo "" >> $HASHFILE

# Hash the code
echo "Analysis code:" >> $HASHFILE
cd code
sha256sum qcd_entropy_code.py >> ../$HASHFILE
cd ..
sha256sum simple_extract_figures.py >> $HASHFILE
echo "" >> $HASHFILE

# Hash data files
echo "Data files:" >> $HASHFILE
cd data
sha256sum *.csv >> ../$HASHFILE
cd ..
echo "" >> $HASHFILE

# Create a master hash of all hashes
echo "Generating master hash..."
MASTER_HASH=$(sha256sum $HASHFILE | cut -d' ' -f1)
echo "" >> $HASHFILE
echo "Master hash of this file: $MASTER_HASH" >> $HASHFILE

echo "SHA256 hashes saved to $HASHFILE"
echo "Master hash: $MASTER_HASH"
