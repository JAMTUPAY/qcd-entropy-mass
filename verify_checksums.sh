#!/bin/bash
echo "Verifying checksums..."
shasum -a 256 -c SHA256SUMS
