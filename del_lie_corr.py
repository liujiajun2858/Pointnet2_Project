# 定义输入和输出文件的路径
input_file_path = 'D:\\rapea.txt'
output_file_path = 'D:\\rapea_change.txt'

# 使用with语句打开文件，确保文件正确关闭
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
        open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 遍历输入文件的每一行
    for line in input_file:
        # 使用split()方法按空格分割行（如果列是由其他字符分隔的，请更改此处的分隔符）b_change
        columns = line.strip().split()

        # 如果列数足够多（即至少有7列），则删除第7列
        if len(columns) >=7 :
            del columns[6]  # 删除索引为6的列，即第7列

        # 将处理后的列重新组合成字符串，并写入输出文件
        output_file.write(' '.join(columns) + '\n')

print(f"文件处理完成，结果已保存至 {output_file_path}")