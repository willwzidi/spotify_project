import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

client_id = 'fc668f7d3ed94d9f9d0f4f8e9c8af9ad'
client_secret = 'feb9efe5c0224e92a44c52e22aa5b9a2'

# 设置客户端凭证管理器
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

category = [
    "Arts and Entertainment", "Books", "Celebrities", "Comedy", "Design", "Fiction", "Film", "Literature", "Pop Culture", "Culture",
    "Stories", "TV", "Business", "Technology", "Careers", "Economics", "Finance", "Marketing", "Educational", "Government",
    "History", "Language", "Philosophy", "Science", "Games", "Video Games", "Beauty", "Fashion", "Fitness and Nutrition",
    "Food", "Health", "Hobbies", "Lifestyle", "Meditation", "Parenting", "Relationship", "Self-care", "Sex", "News", "Politics",
    "Baseball", "Basketball", "Boxing", "American Football", "Hockey", "MMA", "Outdoor", "Rugby", "Running", "Soccer", "Tennis",
    "Wrestling", "True Crime", "Sports News"
]

# 初始化一些变量
collected_podcasts = []
collected_podcast_ids = set()
total = 0
# 遍历每个类别，直到 total_episodes 大于 20000 为止
for term in category:
    total_episodes = 0
    results = sp.search(q=term, type="show", limit=50)
    while results:
        shows = results['shows']['items']
        for show in shows:
            if show['id'] not in collected_podcast_ids and show['total_episodes'] <= 2000:
                total_episodes += show['total_episodes']
                collected_podcasts.append({
                    'name': show['name'],
                    'id': show['id'],
                    'description': show['description'],
                    'category': term
                })
                collected_podcast_ids.add(show['id'])
            # 如果集数超过 20000，停止搜索
            if total_episodes > 2500:
                print(term, total_episodes)
                break
        if total_episodes > 2500:
            total += total_episodes
            print(term, total_episodes)
            break
        # 获取下一页
        if results['shows']['next']:
            results = sp.next(results['shows'])
        else:
            break
print(total)

# 将收集到的 Podcast 信息写入 CSV 文件
with open('podcast_results.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'ID', 'Description', 'category'])  # 写入表头
    for podcast in collected_podcasts:
        writer.writerow([podcast['name'], podcast['id'], podcast['description'], podcast['category']])

print(f"搜索完成，共收集到 {len(collected_podcasts)} 个 Podcast。")
