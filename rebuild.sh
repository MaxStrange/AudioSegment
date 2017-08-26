rm -rf build/
rm -rf dist/
rm -rf audiosegment.egg-info/
pipreqs --force .
python3 setup.py bdist_wheel
twine upload dist/*
echo "Done. Please remember to make a release on github."
