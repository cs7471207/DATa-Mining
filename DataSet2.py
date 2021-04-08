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

def Using_Most_Fre(points):
    mode = points.mode()
    for i in points:
        if(pd.isna(i)):
            i = mode
    return points

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


def Using_Similarity(data):

    Location = data['Location']
    Area_Id = data['Area Id']
    Beat = data['Beat']
    Priority = data['Priority']
    Incident_Type_Id = data['Incident Type Id']
    duration = data['Duration']

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
    Location_Dist = Create_Dict(Location, data)
    Area_Id_Dist = Create_Dict(Area_Id, data)
    Beat_Dist = Create_Dict(Beat, data)
    Priority_Dist = Create_Dict(Priority, data)
    Incident_Type_Id_Dist = Create_Dict(Incident_Type_Id, data)


    Location_Dist_mode = Find_Dict_Mode(Location_Dist)
    Area_Id_Dist_mode = Find_Dict_Mode(Area_Id_Dist)
    Beat_Dist_mode = Find_Dict_Mode(Beat_Dist)
    Priority_Dist_mode = Find_Dict_Mode(Priority_Dist)
    Incident_Type_Id_Dist_mode = Find_Dict_Mode(Incident_Type_Id_Dist)

    duration_1 = duration

    for i in range(duration.shape[0]):
        n=0
        coun_1 = 0
        prov_1 = 0
        reg_1_1 = 0
        reg_2_1 = 0
        var_1 = 0
        wine_1 = 0
        pri_1 = 0
        #print(points[i],i)

        if(pd.isna(duration.iloc[i])):
            a = data.iloc[i]
            print(i)
            if((1-pd.isna(a['Location']))
                and (1-(Location_Dist_mode.get(a['Location']) is None))
                    and (1-Location_Dist_mode.get(a['Location'])[5].empty)
                        and (1-pd.isna(Location_Dist_mode.get(a['Location'])[5][0]))):
                coun_1 = Location_Dist_mode.get(a['Location'])[5][0]
                n = n + 1

            if((1-pd.isna(a['Area Id']))
                and (1-(Area_Id_Dist_mode.get(a['Area Id']) is None))
                    and (1-Area_Id_Dist_mode.get(a['Area Id'])[5].empty)
                        and (1-pd.isna(Area_Id_Dist_mode.get(a['Area Id'])[5][0]))):
                prov_1 = Area_Id_Dist_mode.get(a['Area Id'])[5][0]
                n = n + 1

            if((1-pd.isna(a['Beat']))
                and (1-(Beat_Dist_mode.get(a['Beat']) is None))
                    and (1-Beat_Dist_mode.get(a['Beat'])[5].empty)
                        and (1-pd.isna(Beat_Dist_mode.get(a['Beat'])[5][0]))):
                reg_1_1 = Beat_Dist_mode.get(a['Beat'])[5][0]
                n = n + 1

            if((1-pd.isna(a['Priority']))
                and (1-(Priority_Dist_mode.get(a['Priority']) is None))
                    and (1-Priority_Dist_mode.get(a['Priority'])[5].empty)
                        and (1-pd.isna(Priority_Dist_mode.get(a['Priority'])[5][0]))):
                reg_2_1 = Priority_Dist_mode.get(a['Priority'])[5][0]
                n = n + 1

            if((1-pd.isna(a['Incident Type Id']))
                and (1-(Incident_Type_Id_Dist_mode.get(a['Incident Type Id']) is None))
                    and (1-Incident_Type_Id_Dist_mode.get(a['Incident Type Id'])[5].empty)
                        and (1-pd.isna(Incident_Type_Id_Dist_mode.get(a['Incident Type Id'])[5][0]))):
                var_1 = Incident_Type_Id_Dist_mode.get(a['Incident Type Id'])[5][0]
                n = n + 1

            if(n == 0):
                n = 1
            duration_1[i] = (coun_1+prov_1+reg_1_1+reg_2_1+var_1)/n
            #print(points_1[i])

    return duration_1

def Find_Dict_Mode(Dict):
    Dict_Mode = {}
    print("Finding Mode")
    for key in Dict.keys():
        data = Dict[key]
        data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
        Location = data['Location'].dropna(axis=0, how='any')
        Area_Id = data['Area Id'].dropna(axis=0, how='any')
        Beat = data['Beat'].dropna(axis=0, how='any')
        Priority = data['Priority'].dropna(axis=0, how='any')
        Incident_Type_Id = data['Incident Type Id'].dropna(axis=0, how='any')
        Duration = data['Duration'].dropna(axis=0, how='any')

        Dict_Mode[key] = [Location.mode(),
                                   Area_Id.mode(),
                                   Beat.mode(),
                                   Priority.mode(),
                                   Incident_Type_Id.mode(),
                                   Duration.mode()]

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

def Find_Most_Similarity(data):
    Duration_1 = data['Duration']
    Duration = Duration_1

    for i in range(Duration_1.shape[0]):
        if (pd.isna(Duration_1.iloc[i])):
            source = data.iloc[i]
            temp_unsimilarity = 0
            target_index = -1
            sum_unsimilarity = 1000

            for j in range(data.shape[0]):
            #for j in range(10):
                if i != j:
                    dist = data.iloc[j]
                    if((1-pd.isna(source['Location'])) and (1-pd.isna(dist['Location']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['Location'],dist['Location'])
                    if(pd.isna(source['Location']) and pd.isna(dist['Location'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['Area Id'])) and (1-pd.isna(dist['Area Id']))):
                        temp_unsimilarity = temp_unsimilarity + Number_UnSimilarity(source['Area Id'],dist['Area Id'])
                    if(pd.isna(source['Area Id']) and pd.isna(dist['Area Id'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['Beat'])) and (1-pd.isna(dist['Beat']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['Beat'],dist['Beat'])
                    if(pd.isna(source['Beat']) and pd.isna(dist['Beat'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['Priority'])) and (1-pd.isna(dist['Priority']))):
                        temp_unsimilarity = temp_unsimilarity + Number_UnSimilarity(source['Priority'],dist['Priority'])
                    if(pd.isna(source['Priority']) and pd.isna(dist['Priority'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if((1-pd.isna(source['Incident Type Id'])) and (1-pd.isna(dist['Incident Type Id']))):
                        temp_unsimilarity = temp_unsimilarity + Nominal_UnSimilarity(source['Incident Type Id'],dist['Incident Type Id'])
                    if(pd.isna(source['Incident Type Id']) and pd.isna(dist['Incident Type Id'])):
                        temp_unsimilarity = temp_unsimilarity + 1

                    if(temp_unsimilarity < sum_unsimilarity):
                        sum_unsimilarity = temp_unsimilarity
                        target_index = j

            if(1-pd.isna(data.iloc[target_index]['Duration'])):
                Duration[i] = data.iloc[target_index]['Duration']

    return Duration

def Five_Number(olddata):
    m = olddata.shape[0]
    data = olddata.dropna(axis=0, how='all')
    n = data.shape[0]
    data = sorted(data ,reverse = False)

    print('min', data[0])
    print('Q1',data[int(n/4)])
    print('Q2',data[int(n/2)])
    print('Q3',data[int(n*3/4)])
    print('max',data[n-1])
    print('空缺值个数',m-n)

index_list = [0,1,2,5,6,7,8,9]
data = pd.read_csv("records-for-2011.csv",low_memory=False,encoding="UTF-8",usecols=[1,2,3,4,5,6,7,9])

#data = pd.concat([data_1,data_2])
data = data.drop_duplicates(subset=None, keep='first', inplace=False)
time = pd.to_datetime(data['Create Time'])
month = pd.to_datetime(data['Create Time']).dt.month
day = pd.to_datetime(data['Create Time']).dt.day

StartTime = pd.to_datetime(data['Create Time'])
EndTime = pd.to_datetime(data['Closed Time'])



duration = (EndTime - StartTime)
for i in range(len(duration)):
    duration[i] = duration[i].seconds

temp = pd.Series(duration,name='Duration',index=data.index)
data = pd.concat([data,temp],axis=1)

#duration = Using_Similarity(data)
#duration = Find_Most_Similarity(data)

#plot_box(duration,False)
#plot_hit(duration,False)

Location = data['Location']
Area_Id = data['Area Id']
Beat = data['Beat']
Priority = data['Priority']
Incident_Type_Id = data['Incident Type Id']
Duration = data['Duration']

'''
plot_scatter(Location,Duration,True)
plot_scatter(Area_Id,Duration,True)
plot_scatter(Beat,Duration,True)
plot_scatter(Priority,Duration,True)
plot_scatter(Incident_Type_Id,Duration,True)
'''

Location_fre = Location.value_counts()
Location_fre.to_csv("Location.csv")

Area_Id_fre = Area_Id.value_counts()
Area_Id_fre.to_csv("Area_Id.csv")

Beat_fre = Beat.value_counts()
Beat_fre.to_csv("Beat.csv")

Priority_fre = Priority.value_counts()
Priority_fre.to_csv("Priority.csv")

Incident_Type_Id_fre = Incident_Type_Id.value_counts()
Incident_Type_Id_fre.to_csv("Incident_Type_Id.csv")