import pandas as pd
from mygene import MyGeneInfo

# 1. 读取数据
df = pd.read_csv('../gnomad.v4.1.constraint_metrics.tsv', sep='\t')  # 修改为你的文件名和分隔符

# 2. 统一gene_id为ENSEMBL ID
mg = MyGeneInfo()
gene_ids = df['gene_id'].astype(str).tolist()
query = mg.querymany(gene_ids, scopes=['ensembl.gene', 'symbol'], fields='ensembl.gene', species='human')
id_map = {}
for item in query:
    if 'ensembl' in item:
        if isinstance(item['ensembl'], list):
            id_map[item['query']] = item['ensembl'][0]['gene']
        else:
            id_map[item['query']] = item['ensembl']['gene']
    else:
        id_map[item['query']] = None

df['ensembl_id'] = df['gene_id'].map(id_map)
df = df.dropna(subset=['ensembl_id'])

# 3. 筛选mane_select为true
df = df[df['canonical'] == True]

# 4. 提取lof.oe_ci.upper列，并删除缺失值
result = df[['ensembl_id', 'lof.oe_ci.upper']].dropna()

# 5. 输出txt文件
result.to_csv('ensembl_loeuf.txt', sep='\t', index=False, header=False)
