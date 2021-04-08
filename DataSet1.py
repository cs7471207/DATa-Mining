import pandas as pd
import numpy as np
from math import isnan
import matplotlib.pyplot as plt

def winnow_nominal_attribute(check_str):
    #check whether stringn and nan
    if(1-isinstance(check_str,str)):
        if (isnan(check_str)):
            return False;
        return True;

    #check string contains chinese
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fa5':
            return True
    return False

def number_analysis(points):
    points = points.dropna(axis=0,how='all')
    sort_points = sorted(points)

    min = sort_points[0]
    Q1 = sort_points[int(points.shape[0] / 4)]
    Q2 = sort_points[int(points.shape[0] / 2)]
    Q3 = sort_points[int(points.shape[0] * 3 / 4)]
    max = sort_points[points.shape[0] - 1]
    return min,Q1,Q2,Q3,max

def plot_box(points, cond):
    if(cond):
        points = points.dropna(axis=0, how='all')

    df = pd.DataFrame(points)
    df.plot.box(title="BoxPlot")
    plt.grid(linestyle="--", alpha=0.1)
    plt.show()

def plot_hit(points, cond):
    if(cond):
        points = points.dropna(axis=0, how='all')
    plt.hist(points)
    plt.show()

def plot_scatter(a_1, a_2, cond):
    data = pd.concat([a_1, a_2],axis = 1)
    if(cond):
        data = data.dropna(axis = 0, how = 'any')

    a_1 = data.iloc[:,0]
    a_2 = data.iloc[:,1]
    plt.scatter(a_1, a_2)
    plt.xticks(rotation=90)
    plt.show()

def Create_Dict(attribute, data):
    Dict = {}
    for i in range(len(attribute)):
        print(attribute.iloc[i],i)
        if(pd.isna(attribute.iloc[i])):
            continue
        if(attribute.iloc[i] in Dict):
            Dict[attribute.iloc[i]] = pd.concat([Dict.get(attribute.iloc[i]),data.iloc[i].to_frame()],ignore_index=True,axis=1)
        else:
            Dict[attribute.iloc[i]] = data.iloc[i].to_frame()
    return Dict

def Using_Most_Fre(points):
    mode = points.mode()
    for i in points:
        if(pd.isna(i)):
            i = mode
    return points

def Using_Similarity(data):

    country = data['country']
    points = data['points']
    price = data['price']
    province = data['province']
    region_1 = data['region_1']
    region_2 = data['region_2']
    variety = data['variety']
    winery = data['winery']
    '''

    country = data['country'].iloc[0:5000]
    points = data['points'].iloc[0:5000]
    price = data['price'].iloc[0:5000]
    province = data['province'].iloc[0:5000]
    region_1 = data['region_1'].iloc[0:5000]
    region_2 = data['region_2'].iloc[0:5000]
    variety = data['variety'].iloc[0:5000]
    winery = data['winery'].iloc[0:5000]
        '''
    country_Dist = Create_Dict(country, data)
    province_Dist = Create_Dict(province, data)
    region_1_Dist = Create_Dict(region_1, data)
    region_2_Dist = Create_Dict(region_2, data)
    variety_Dist = Create_Dict(variety, data)
    winery_Dist = Create_Dict(winery, data)
    points_Dist = Create_Dict(points, data)
    price_Dist = Create_Dict(price, data)


    country_Dist_mode = Find_Dict_Mode(country_Dist)
    province_Dist_mode = Find_Dict_Mode(province_Dist)
    region_1_Dist_mode = Find_Dict_Mode(region_1_Dist)
    region_2_Dist_mode = Find_Dict_Mode(region_2_Dist)
    variety_Dist_mode = Find_Dict_Mode(variety_Dist)
    winery_Dist_mode = Find_Dict_Mode(winery_Dist)
    points_Dist_mode = Find_Dict_Mode(points_Dist)
    price_Dist_mode = Find_Dict_Mode(price_Dist)

    points_1 = points
    price_1 = price

    for i in range(points.shape[0]):
        n=0
        coun_1 = 0
        prov_1 = 0
        reg_1_1 = 0
        reg_2_1 = 0
        var_1 = 0
        wine_1 = 0
        pri_1 = 0

        #print(points[i],i)

        if(pd.isna(points.iloc[i])):
            a = data.iloc[i]
            print(i)
            if((1-pd.isna(a['country']))
                and (1-(country_Dist_mode.get(a['country']) is None))
                    and (1-country_Dist_mode.get(a['country'])[6].empty)
                        and (1-pd.isna(country_Dist_mode.get(a['country'])[6][0]))):
                coun_1 = country_Dist_mode.get(a['country'])[6][0]
                n = n + 1

            if((1-pd.isna(a['province']))
                and (1-(province_Dist_mode.get(a['province']) is None))
                    and (1-province_Dist_mode.get(a['province'])[6].empty)
                        and (1-pd.isna(province_Dist_mode.get(a['province'])[6][0]))):
                prov_1 = province_Dist_mode.get(a['province'])[6][0]
                n = n + 1

            if((1-pd.isna(a['region_1']))
                and (1-(region_1_Dist_mode.get(a['region_1']) is None))
                    and (1-region_1_Dist_mode.get(a['region_1'])[6].empty)
                        and (1-pd.isna(region_1_Dist_mode.get(a['region_1'])[6][0]))):
                reg_1_1 = region_1_Dist_mode.get(a['region_1'])[6][0]
                n = n + 1

            if((1-pd.isna(a['region_2']))
                and (1-(region_2_Dist_mode.get(a['region_2']) is None))
                    and (1-region_2_Dist_mode.get(a['region_2'])[6].empty)
                        and (1-pd.isna(region_2_Dist_mode.get(a['region_2'])[6][0]))):
                reg_2_1 = region_2_Dist_mode.get(a['region_2'])[6][0]
                n = n + 1

            if((1-pd.isna(a['variety']))
                and (1-(variety_Dist_mode.get(a['variety']) is None))
                    and (1-variety_Dist_mode.get(a['variety'])[6].empty)
                        and (1-pd.isna(variety_Dist_mode.get(a['variety'])[6][0]))):
                var_1 = variety_Dist_mode.get(a['variety'])[6][0]
                n = n + 1

            if((1-pd.isna(a['winery']))
                and (1-(winery_Dist_mode.get(a['winery']) is None))
                    and (1-winery_Dist_mode.get(a['winery'])[6].empty)
                        and (1-pd.isna(winery_Dist_mode.get(a['winery'])[6][0]))):
                wine_1 = winery_Dist_mode.get(a['winery'])[6][0]
                n = n + 1

            if((1-pd.isna(a['price']))
                and (1-(price_Dist_mode.get(a['price']) is None))
                    and (1-points_Dist_mode.get(a['price'])[6].empty)
                        and (1-pd.isna(points_Dist_mode.get(a['price'])[6][0]))):
                pri_1 = points_Dist_mode.get(a['price'])[6][0]
                n = n + 1
            if(n == 0):
                n = 1
            points_1[i] = (coun_1+prov_1+reg_1_1+reg_2_1+var_1+wine_1+pri_1)/n
            #print(points_1[i])

    for i in range(price_1.shape[0]):
        n=0
        coun = 0
        prov = 0
        reg_1 = 0
        reg_2 = 0
        var = 0
        wine = 0
        pri = 0

        if(pd.isna(price_1.iloc[i])):
            a = data.iloc[i]

            if((1-pd.isna(a['country']))
                and (1-(country_Dist_mode.get(a['country']) is None))
                    and (1-country_Dist_mode.get(a['country'])[7].empty)
                        and (1-pd.isna(country_Dist_mode.get(a['country'])[7][0]))):
                coun = country_Dist_mode.get(a['country'])[7][0]
                n = n + 1

            if((1-pd.isna(a['province']))
                and (1-(province_Dist_mode.get(a['province']) is None))
                    and (1-province_Dist_mode.get(a['province'])[7].empty)
                        and (1-pd.isna(province_Dist_mode.get(a['province'])[7][0]))):
                prov = province_Dist_mode.get(a['province'])[7][0]
                n = n + 1

            if((1-pd.isna(a['region_1']))
                and (1-(region_1_Dist_mode.get(a['region_1']) is None))
                    and (1-region_1_Dist_mode.get(a['region_1'])[7].empty)
                        and (1-pd.isna(region_1_Dist_mode.get(a['region_1'])[7][0]))):
                reg_1 = region_1_Dist_mode.get(a['region_1'])[7][0]
                n = n + 1

            if((1-pd.isna(a['region_2']))
                and (1-(region_2_Dist_mode.get(a['region_2']) is None))
                    and (1-region_2_Dist_mode.get(a['region_2'])[7].empty)
                        and (1-pd.isna(region_2_Dist_mode.get(a['region_2'])[7][0]))):
                reg_2 = region_2_Dist_mode.get(a['region_2'])[7][0]
                n = n + 1

            if((1-pd.isna(a['variety']))
                and (1-(variety_Dist_mode.get(a['variety']) is None))
                        and (1-variety_Dist_mode.get(a['variety'])[7].empty)
                            and (1-pd.isna(variety_Dist_mode.get(a['variety'])[7][0]))):
                var = variety_Dist_mode.get(a['variety'])[7][0]
                n = n + 1

            if((1-pd.isna(a['winery']))
                and (1-(winery_Dist_mode.get(a['winery']) is None))
                    and (1-winery_Dist_mode.get(a['winery'])[7].empty)
                        and (1-pd.isna(winery_Dist_mode.get(a['winery'])[7][0]))):
                wine = winery_Dist_mode.get(a['winery'])[7][0]
                n = n + 1

            if((1-pd.isna(a['points']))
                and (1-(points_Dist_mode.get(a['points']) is None))
                    and (1-points_Dist_mode.get(a['points'])[7].empty)
                        and (1-pd.isna(points_Dist_mode.get(a['points'])[7][0]))):
                pri = points_Dist_mode.get(a['points'])[7][0]
                n = n + 1

            if(n == 0):
                n = 1
            price_1[i] = (coun+prov+reg_1+reg_2+var+wine+pri)/n
            #print(price_1[i])
    return points_1,price_1

def Find_Dict_Mode(Dict):
    Dict_Mode = {}
    print("Finding Mode")
    for key in Dict.keys():
        data = Dict[key]
        data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
        country = data['country'].dropna(axis=0, how='any')
        points = data['points'].dropna(axis=0, how='any')
        price = data['price'].dropna(axis=0, how='any')
        province = data['province'].dropna(axis=0, how='any')
        region_1 = data['region_1'].dropna(axis=0, how='any')
        region_2 = data['region_2'].dropna(axis=0, how='any')
        variety = data['variety'].dropna(axis=0, how='any')
        winery = data['winery'].dropna(axis=0, how='any')

        Dict_Mode[key] = [country.mode(),
                                   province.mode(),
                                   region_1.mode(),
                                   region_2.mode(),
                                   variety.mode(),
                                 winery.mode(),
                                   points.mode(),
                             price.mode()]

    print("Ending Finding Mode")
    return Dict_Mode

def levenshtein(first, second):
    if len(first) > len(second):
        first, second = second, first
    if len(first) == 0:
        return len(second)
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first[i - 1] != second[j - 1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
            # print distance_matrix
    Dist = distance_matrix[first_length - 1][second_length - 1]
    del distance_matrix
    return Dist

def Nominal_UnSimilarity(s1,s2):
    m = levenshtein(s1,s2)
    return m

def Number_UnSimilarity(n1,n2):
    Max = max(n1,n2)
    Min = min(n1,n2)
    return Max-Min

def Find_Most_Similarity(olddata):
    points_1 = olddata['points'].iloc[0:10000]
    price_1 = olddata['price'].iloc[0:10000]
    data = olddata.iloc[0:10000]
    price = price_1
    points = points_1

    for i in range(points_1.shape[0]):
        if (pd.isna(points_1.iloc[i])):
            source = data.iloc[i]
            temp_unsimilarity = 0
            target_index = -1
            sum_unsimilarity = 1000

            for j in range(data.shape[0]):
            #for j in range(10):
                if i != j:
                    dist = data.iloc[j]
                    if((1-pd.isna(source['country'])) and (1-pd.isna(dist['country']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['country'],dist['country'])
                    if(pd.isna(source['country']) and pd.isna(dist['country'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['province'])) and (1-pd.isna(dist['province']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['province'],dist['province'])
                    if(pd.isna(source['province']) and pd.isna(dist['province'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['region_1'])) and (1-pd.isna(dist['region_1']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['region_1'],dist['region_1'])
                    if(pd.isna(source['region_1']) and pd.isna(dist['region_1'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['region_2'])) and (1-pd.isna(dist['region_2']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['region_2'].iloc[0],dist['region_2'])
                    if(pd.isna(source['region_2']) and pd.isna(dist['region_2'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['variety'])) and (1-pd.isna(dist['variety']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['variety'],dist['variety'])
                    if(pd.isna(source['variety']) and pd.isna(dist['variety'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['winery'])) and (1-pd.isna(dist['winery']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['winery'],dist['winery'])
                    if(pd.isna(source['winery']) and pd.isna(dist['winery'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['price'])) and (1-pd.isna(dist['price']))):
                        temp_unsimilarity = temp_unsimilarity + Number_UnSimilarity(source['price'],dist['price'])
                    if(pd.isna(source['price']) and pd.isna(dist['price'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if(temp_unsimilarity < sum_unsimilarity):
                        sum_unsimilarity = temp_unsimilarity
                        target_index = j

            if(1-pd.isna(data.iloc[target_index]['points'])):
                points[i] = data.iloc[target_index]['points']

    for i in range(price_1.shape[0]):
        if (pd.isna(price_1.iloc[i])):
            source = data.iloc[i]
            temp_unsimilarity = 0
            temp_index = 0
            target_index = 0
            sum_unsimilarity = 1000
            print(i)
            for j in range(data.shape[0]):
            #for j in range(10):
                if i != j:
                    dist = data.iloc[j]
                    if((1-pd.isna(source['country'])) and (1-pd.isna(dist['country']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['country'],dist['country'])
                    if(pd.isna(source['country']) and pd.isna(dist['country'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['province'])) and (1-pd.isna(dist['province']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['province'],dist['province'])
                    if(pd.isna(source['province']) and pd.isna(dist['province'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['region_1'])) and (1-pd.isna(dist['region_1']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['region_1'],dist['region_1'])
                    if(pd.isna(source['region_1']) and pd.isna(dist['region_1'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['region_2'])) and (1-pd.isna(dist['region_2']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['region_2'],dist['region_2'])
                    if(pd.isna(source['region_2']) and pd.isna(dist['region_2'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['variety'])) and (1-pd.isna(dist['variety']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['variety'],dist['variety'])
                    if(pd.isna(source['variety']) and pd.isna(dist['variety'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['winery'])) and (1-pd.isna(dist['winery']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['winery'],dist['winery'])
                    if(pd.isna(source['winery']) and pd.isna(dist['winery'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['points'])) and (1-pd.isna(dist['points']))):
                        temp_unsimilarity = temp_unsimilarity + Number_UnSimilarity(source['points'],dist['points'])
                    if(pd.isna(source['points']) and pd.isna(dist['points'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if(temp_unsimilarity < sum_unsimilarity):
                        sum_unsimilarity = temp_unsimilarity
                        target_index = j
            if(1-pd.isna(data.iloc[target_index]['price'])):
                price[i] = data.iloc[target_index]['price']

    return points,price

index_list = [0,1,2,5,6,7,8,9]
data_1 = pd.read_csv("wine-1.csv",low_memory=False,encoding="UTF-8",usecols=[1,2,3,4,5,6,7,8,9,10])
data_2 = pd.read_csv("wine-2.csv",low_memory=False,encoding="UTF-8",usecols=[1,2,3,4,5,6,7,8,9,10])
data = pd.concat([data_1,data_2])
data = data.drop_duplicates(subset=None, keep='first', inplace=False)

#points,price = Using_Similarity(data)
points,price = Find_Most_Similarity(data)

plot_box(points,False)
plot_hit(points,False)
plot_box(price,False)
plot_hit(price,False)

#points.to_csv("NewPoints.csv")
#price.to_csv("NewPrice.csv")

points.to_csv("ObjectPoints.csv")
price.to_csv("ObjectPrice.csv")
'''
country_Dist = Create_Dict(country,data)
country_fre = country.value_counts()
country_fre.to_csv("country.csv")

designation_Dist = Create_Dict(designation,data)
designation_fre = designation.value_counts()
designation_fre.to_csv("designation.csv")

province_Dist = Create_Dict(province,data)
province_fre = province.value_counts()
province_fre.to_csv("province.csv")

region_1_Dist = Create_Dict(region_1,data)
region_1_fre = region_1.value_counts()
region_1_fre.to_csv("region_1.csv")

region_2_Dist = Create_Dict(region_2,data)
region_2_fre = region_2.value_counts()
region_2_fre.to_csv("region_2.csv")

variety_Dist = Create_Dict(variety,data)
variety_fre = variety.value_counts()
variety_fre.to_csv("variety.csv")

winery_Dist = Create_Dict(winery,data)
winery_fre = winery.value_counts()
winery_fre.to_csv("winery.csv")

points_Dist = Create_Dict(points,data)

price_Dist = Create_Dict(price,data)

'''
