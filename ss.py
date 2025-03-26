import requests

response = requests.get("https://api.tavily.com/search", verify=False)
print(response.content)
