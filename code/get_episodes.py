import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import pandas as pd

# 替换为您的 Client ID 和 Client Secret
# client_id = '2ed41a45eaab4d37b163f990a2e638a4'
# client_secret = '087b93336cfc4d7a940a519be3d8e7ad'
client_id = 'fc668f7d3ed94d9f9d0f4f8e9c8af9ad'
client_secret = 'feb9efe5c0224e92a44c52e22aa5b9a2'

# 设置客户端凭证管理器
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

csv_file = "podcast_results.csv"
podcast_data = pd.read_csv(csv_file)
podcast_data = podcast_data

# 保存的目标文件
output_file = "podcast_episodes.csv"

# 创建并初始化 CSV 文件，写入表头
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Podcast_ID", "Episodes_ID", "Podcast_name", "Episodes_name", "category", "podcast_description", "Description"])
total = 0
# 使用 iterrows 遍历每一行
for index, row in podcast_data.iterrows():
    Podcast_name = row['Name']
    podcast_id = row['ID']
    podcast_description = row['Description']
    category = row['category']

    # 分页获取所有单集
    next_page = sp.show_episodes(podcast_id, limit=50)  # 初始获取
    episode_list = []

    while next_page:
        for episode in next_page['items']:
            if episode is None:  # 跳过无效数据
                continue

            # 获取字段并设置默认值
            Episodes_id = episode.get('id', pd.NA)
            title = episode.get('name', pd.NA)
            description = episode.get('description', pd.NA)

            # 添加到待写入列表中
            episode_list.append([podcast_id, Episodes_id, Podcast_name, title, category, podcast_description, description])

            # # 当积累到一定数量时保存到文件
            # if len(episode_list) >= 1000:
            #     # 使用 'a' 模式将数据追加到文件中
            #     with open(output_file, "a", newline="", encoding="utf-8") as file:
            #         writer = csv.writer(file)
            #         writer.writerows(episode_list)
            #     total += len(episode_list)
            #     episode_list = []  # 清空列表

        # 获取下一页数据
        next_page = sp.next(next_page)

    # 保存剩余的数据
    if episode_list:
        with open(output_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(episode_list)
        total += len(episode_list)
        print(total)

print("sucssesfully saved")
