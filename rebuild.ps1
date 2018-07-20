Remove-Item build -Recurse -Force
Remove-Item dist -Recurse -Force
Remove-Item audiosegment.egg-info -Recurse -Force

pipreqs --force .

python .\build_the_docs.py
python setup.py bdist_wheel

twine upload dist/*

Write-Host "Done. Please remember to make a release on github via:"
Write-Host "git tag -a v<VERSION_NUMBER> -m <MSG>"
Write-Host "git push origin v<VERSION_NUMBER>"

