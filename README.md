# DATa-Mining
数据集说明：
选取了两个数据集：Wine Reviews 和 Oakland Crime Statistics 2011 to 2016

函数说明：
1.winnow_nominal_attribute(check_str)
    检查英文的标称属性中是否含有中文乱码

2.number _analysis(attribute)
    对输入的属性进行五数概括

3.plot_box(attribute，cond）
    对该属性绘制盒型图，cond用于判断是否需要去掉attribute中的nan值

4.plot_hit(attribute，cond)
    对该属性绘制直方图，cond用于判断是否需要去掉attribute中的nan值

5.plot_scatter(attribute_1，attribute_2，cond）
    绘制attribute_1，attribute_2之间的散点图，cond用于判断是否需要去掉attribute中的nan值

6.Using_Most_Fre(attribute)
    计算attribute中的众数，并用该众数填充attribute中的缺失值，返回填充后的attribute

7.Create_Dict(attribute，data)
    将数据集data按照attribute进行分类，返回分类后的字典

8.Find_Dict_Mode(Dict)
    计算Create_Dict返回结果中，每个分类各个属性的众数，返回记录众数的字典

9.Using_Similarity(data)
     对于要填充的属性attribute，根据其在data数据集中的位置找到其在数据集中对应的元组，从该元组中提取出每个属性的取值，分别在Find_Dict_Mode返回的字典中找到该取值对应的在attribute属性上的众数，求出他们的平均值作为填充结果。

10.levenshtein(first，second)
    计算两个字符串之间的最少修改距离，用于衡量标称属性之间的距离

11.Number_UnSimilarity(n1,n2)
    计算两个数值属性之间的L1距离，用于衡量数值属性之间的距离

12.Find_Most_Similarity(data)
    找出要填充数据在data数据集中对应元组与data数据集中最相近的对象，以该对象的对应属性值作为填充值填充该属性，对每个所选属性都进行距离计算，找出距离最小值的对象。
