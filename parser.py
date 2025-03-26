import re
with open('fetch.txt', 'r', encoding='utf-8') as f:
    data = f.read()



names = re.findall(r'"author":{"name":"(.*?)"', data)
comments = re.findall(r'\},"text":"(.*?)","textLanguage":',data)
ratings = re.findall(r'"rating":(.)',data)
ratings = [int(i) for i in ratings]
print(len(names),len(comments),len(ratings))
file = open('result.txt','w',encoding='utf-8')
for i in range(min(len(names),len(comments),len(ratings))):
    file.write(f'{comments[i]}/{ratings[i]}/{names[i]} \n')
file.close()