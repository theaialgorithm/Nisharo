#!/bin/bash
# Download the public-domain English training corpus (no API needed).
# Run this before prepare_data.py to populate data/*.txt

mkdir -p data
cd data

echo "Downloading training corpus from Project Gutenberg..."

curl -sSL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -o shakespeare.txt
echo "  shakespeare.txt     $(wc -c < shakespeare.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/1342/1342-0.txt" -o pride_prejudice.txt
echo "  pride_prejudice.txt $(wc -c < pride_prejudice.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/1661/1661-0.txt" -o sherlock.txt
echo "  sherlock.txt        $(wc -c < sherlock.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/11/11-0.txt" -o alice.txt
echo "  alice.txt           $(wc -c < alice.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/84/84-0.txt" -o frankenstein.txt
echo "  frankenstein.txt    $(wc -c < frankenstein.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/98/98-0.txt" -o tale_two_cities.txt
echo "  tale_two_cities.txt $(wc -c < tale_two_cities.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/2701/2701-0.txt" -o moby_dick.txt
echo "  moby_dick.txt       $(wc -c < moby_dick.txt) bytes"

curl -sSL "https://www.gutenberg.org/files/174/174-0.txt" -o dorian_gray.txt
echo "  dorian_gray.txt     $(wc -c < dorian_gray.txt) bytes"

echo ""
echo "Done! Total: $(cat *.txt | wc -c) bytes of English text."
echo "Now run: python prepare_data.py"
