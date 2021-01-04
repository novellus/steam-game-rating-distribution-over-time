import json
import requests
import yaml


# API Docs at https://partner.steamgames.com/doc/webapi/ISteamApps
manifest = json.loads(requests.get('http://api.steampowered.com/ISteamApps/GetAppList/v2/?format=json').text)

f=open('01_steam_manifest.yaml', 'w')
f.write(yaml.dump(manifest))
f.close()
