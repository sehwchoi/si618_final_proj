import json
import itertools
import operator

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, stdev

from scipy.stats import chi2_contingency

sns.set(style='ticks', color_codes=True)

VIDEO_FILE = "video_cache_file.json"
CATEGORY_FILE = "video_category.json"

"""
    code_to_name = {'US': 'USA', 'CA': 'Canada', 'SE': 'Sweden', 'FR': 'France',
                    'KR': 'Korea', 'JP': 'Japan', 'ID': 'Indonesia', 'TW': 'Taiwan'}
"""
WESTERN = ['US', 'CA', 'SE', 'FR']
ASIA = ['KR', 'JP', 'ID', 'TW']

#def reshape_data(video_df):
#    video_pvt_tbl = pd.pivot_table(video_df, index=['code'], )

def find_same_video_rate(df):
    id_by_code_df = df.groupby('code')['id'].apply(list).reset_index(name='ids')
    #print('id_by_code_df', id_by_code_df['code'])
    result = {}
    for combination in itertools.combinations(id_by_code_df['code'], 2):
        code1 = combination[0]
        code2 = combination[1]

        code1_ids = list(id_by_code_df.loc[id_by_code_df['code'] == code1]['ids'])[0]
        code2_ids = list(id_by_code_df.loc[id_by_code_df['code'] == code2]['ids'])[0]

        same_ids = list(set(code1_ids) & set(code2_ids))
        count = len(same_ids)
        rate = float(count / len(code1_ids))
        result[(code1, code2)] = rate

    result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    #print(result)
    pairs = ["{}-{}".format(data[0][0], data[0][1]) for data in result]
    rates = [data[1] for data in result]
    result_df = pd.DataFrame(dict(pair=pairs, rate=rates))
    print(result_df)
    ax = sns.barplot(x="pair", y="rate", data=result_df)
    ax.set(xlabel='country pair', ylabel='rates of same video')
    ax.set_xticklabels(pairs, rotation=45)
    plt.title("Rate of Same Video between Two Countries")
    plt.savefig("same_video_rate.png")

def test_chi_squared(df1, df2):
    df_merged = pd.merge(df1, df2, on='category', how='outer').fillna(0)
    result = chi2_contingency([df_merged['count_x'], df_merged['count_y']])
    print('p-value', result[1])



def analyze_category_id(df):
    pvt_tbl = pd.pivot_table(df, index=['code', 'category'], values=['id'], aggfunc=len, fill_value=0)
    result_tbl = pvt_tbl.reset_index().sort_values(['code', 'id'], ascending=[1, 0]).set_index(['code', 'category'])
    print(result_tbl.index)
    print(result_tbl)

    for country in result_tbl.index.levels[0]:
        data = result_tbl.loc[country]
        #print(data.index, data.values)

        fig1, ax1 = plt.subplots()

        ax1.pie(data.values, labels=data.index, autopct='%1.1f%%',
                shadow=True, startangle=30)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        plt.title("Category_{}".format(country))
        plt.savefig("pie_chart_{}.png".format(country))

    pvt_tbl = pd.pivot_table(df, index=['region', 'category'], values=['id'], aggfunc=len, fill_value=0)
    region_tbl = pvt_tbl.reset_index().sort_values(['region', 'id'], ascending=[1, 0]).set_index(['region', 'category'])
    region_tbl.columns = ['count']
    print(region_tbl)

    # Pie chart
    dfs = []
    for region in ['asia', 'western']:
        data = region_tbl.loc[region]
        dfs.append(data.reset_index())
        #print(data.index, data.values)

        fig1, ax1 = plt.subplots()

        ax1.pie(data.values, labels=data.index, autopct='%1.1f%%',
                shadow=True, startangle=30)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        plt.title("Category_{}".format(region))
        plt.savefig("pie_chart_{}.png".format(region))

    test_chi_squared(dfs[0], dfs[1])


def analyze_restriction(df):
    df = df[['code', 'title', 'restricted']]
    print(df.sort_values(['restricted'], ascending=[0]))

    df_by_code = df.groupby('code')['restricted'].agg('sum').reset_index(name='sum').sort_values(['sum'], ascending=[0])
    print(df_by_code)

    ax = sns.barplot(x="code", y="sum", data=df_by_code)
    ax.set(xlabel='country', ylabel='total number of restricted')
    plt.title("Total Number of Restricted Countries that Each Video Has")
    plt.savefig("restriction_by_country.png")


def analyze_stats(df):
    df_origin = df[['code', 'ratio_dislike_like', 'ratio_like_view']]

    ratio_1_df = df_origin.groupby('code')['ratio_dislike_like'].apply(list).reset_index(name='ratio1')
    for code in WESTERN:
        ratios = list(ratio_1_df.loc[ratio_1_df['code'] == code]['ratio1'])[0]
        ax = sns.kdeplot(ratios, label=code)
        ax.set(xlabel='dislike/like count ratio', ylabel='distribution')
        plt.title("Distribution of Video Dislike/Like Count of Western Countries")
        plt.savefig("dist_ratio_dislike_like_western.png")

    plt.close()
    for code in ASIA:
        ratios = list(ratio_1_df.loc[ratio_1_df['code'] == code]['ratio1'])[0]
        ax = sns.kdeplot(ratios, label=code)
        ax.set(xlabel='dislike/like count ratio', ylabel='distribution')
        plt.title("Distribution of Video Dislike/Like Count of Asia Countries")
        plt.savefig("dist_ratio_dislike_like_asia.png")

    plt.close()
    ratio_2_df = df_origin.groupby('code')['ratio_like_view'].apply(list).reset_index(name='ratio2')
    for code in WESTERN:
        ratios = list(ratio_2_df.loc[ratio_2_df['code'] == code]['ratio2'])[0]
        ax = sns.kdeplot(ratios, label=code)
        ax.set(xlabel='like/view count ratio', ylabel='distribution')
        plt.title("Distribution of Video Like/View Count of Western Countries")
        plt.savefig("dist_ratio_like_view_western.png")

    plt.close()
    for code in ASIA:
        ratios = list(ratio_2_df.loc[ratio_2_df['code'] == code]['ratio2'])[0]
        ax = sns.kdeplot(ratios, label=code)
        ax.set(xlabel='like/view count ratio', ylabel='distribution')
        plt.title("Distribution of Video Like/View Count of Asia Countries")
        plt.savefig("dist_ratio_like_view_asia.png")

    plt.close()
    df = df_origin.groupby('code').agg(['min', 'mean', 'std', 'max']).reset_index()
    print(df)

    df = df_origin.groupby('code').agg(['mean']).reset_index()
    print(df)
    df = pd.melt(df, id_vars="code", var_name="ratio", value_name="ratio_value")
    print(df)
    ax = sns.factorplot(x="code", y="ratio_value", hue="ratio", data=df, kind="bar", aspect=2)
    ax.set(xlabel='country', ylabel='ratios of dislike/like and like/views')
    plt.title("Ratios of Video dislikes/likes and likes/views")
    plt.savefig("ratios.png")


def parse_data():
    global VIDEO_FILE, CATEGORY_FILE, WESTERN, ASIA

    video_dict = {}
    try:
        with open(VIDEO_FILE, 'r') as video_file:
            cache_json = video_file.read()
            video_dict = json.loads(cache_json)
            #print("video", video_dict)
    except:
        print("cannot read video data")

    ctg_dict = {}
    try:
        with open(CATEGORY_FILE, 'r') as category_file:
            cache_json = category_file.read()
            ctg_raw_dict = json.loads(cache_json)

            for data in ctg_raw_dict['items']:
                id = data['id']
                category = data['snippet']['title']
                ctg_dict[id] = category
    except:
        print("cannot read category data")

    codes = []
    ids = []
    titles = []
    categories = []
    regions = []
    restricts = []
    ratios_dislike_like = []
    ratios_like_views = []

    for data in video_dict.values():
        code = data['code']
        region = 'western' if code in WESTERN else 'asia'
        for video in data['resp']:
            id = video['id']
            title = video['snippet']['title']
            category = ctg_dict[video['snippet']['categoryId']]
            try:
                restrict = len(video['contentDetails']['regionRestriction']['blocked'])
            except:
                restrict = 0

            try:
                ratio_1 = float(video['statistics']['dislikeCount']) / float(video['statistics']['likeCount'])
            except:
                ratio_1 = np.nan

            try:
                ratio_2 = float(video['statistics']['likeCount']) / float(video['statistics']['viewCount'])
            except:
                ratio_2 = np.nan
            codes.append(code)
            ids.append(id)
            titles.append(title)
            categories.append(category)
            restricts.append(restrict)
            regions.append(region)
            ratios_dislike_like.append(ratio_1)
            ratios_like_views.append(ratio_2)

    d = {'code': codes, 'region': regions, 'id': ids, 'title': titles,
         'restricted': restricts, 'category': categories,
         'ratio_dislike_like': ratios_dislike_like, 'ratio_like_view': ratios_like_views}

    video_df = pd.DataFrame(data=d)
    print('video_df', video_df)
    video_df.to_csv("video_df.csv", encoding='utf-8')

    return video_df


if __name__ == '__main__':
    df = parse_data()
    #find_same_video_rate(df)
    # analyze_category_id(df)
    # analyze_restriction(df)
    analyze_stats(df)
