set -e

rm -rf build/
rm -rf dist/
rm -rf audiosegment.egg-info/
rm -rf docs/api/
rm -rf algorithms/__pycache__
rm -rf tests/__pycache__
python3 build_the_docs.py
python3 setup.py sdist bdist_wheel
twine upload dist/*
echo "Done. Please remember to make a release on github via:"
echo "git tag -a v<VERSION_NUMBER> -m \"<Message>\""
echo "git push origin v<VERSION_NUMBER>"
