import requests, zipfile, io


with open('rest.txt') as links:
    links = links.read().split('\n')
    for link in links:
        r = requests.get(link, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()