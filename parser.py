import re
import json
with open('fetch.json', 'r', encoding='utf-8') as f:
    data = f.read()



names = re.findall(r'"author":{"name":"(.*?)"', data)
comments = re.findall(r'\},"text":"(.*?)","textLanguage":',data)
ratings = [int(i) for i in re.findall(r'"rating":(.)',data)]

for i in range(min(len(names),len(comments),len(ratings))):
    result = {"names":names,"comments":comments,"ratings":ratings}

    with open('result.json', 'w', encoding='utf-8') as file:
        json.dump(result,file, ensure_ascii = False)
