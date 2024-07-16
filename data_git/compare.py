train_file = 'train.list'
valid_file = 'valid.list'

# 读取train.list文件内容并保存到train_lines列表中
with open(train_file, 'r') as train_f:
    train_lines = train_f.readlines()

# 逐行读取valid.list文件，并与train_lines进行比较
with open(valid_file, 'r') as valid_f:
    valid_lines = valid_f.readlines()

# 删除train.list中与valid.list相同的行
with open(train_file, 'w') as train_f:
    for train_line in train_lines:
        if train_line not in valid_lines:
            train_f.write(train_line)