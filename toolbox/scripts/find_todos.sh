#!/usr/bin/env bash

set -e

echo "Searching TODOs in repository..."
grep -n "TODO" -R . || echo "No TODO found."
