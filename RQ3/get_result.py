import pandas as pd

metric_name_list = ['random', 'kmeans', 'minibatchkmeans','kmeans_plus_plus', 'gaussianmixture',
                    'margin', 'variant_margin', 'variance', 'variant_variance',
                    'leastconfidence', 'variant_leastconfidence', 'deepgini', 'variant_deepgini',
                    'entropy', 'variant_entropy', 'bald', 'nc']


def get_df():
    f = open('predict.log', "r")
    lines = f.readlines()
    lines = [line.strip() for line in lines if len(line) > 5]
    value = [float(i.split('=')[-1].strip()) for i in lines]
    number = [int(i.split('_index_list_')[-1].split('.')[0].strip()) for i in lines]
    metric_name = [i.split('_index_list')[0].strip().lower() for i in lines]

    df = pd.DataFrame(columns=['metric_name'])
    df['metric_name'] = metric_name
    df['number'] = number
    df['value'] = value
    return df


def save_result():
    df = get_df()
    columns_list= ['metric_name', '10%', '20%', '30%', '40%']
    ratio_10_list =[]
    ratio_20_list = []
    ratio_30_list = []
    ratio_40_list = []
    df_re = pd.DataFrame(columns=columns_list)
    for i in metric_name_list:
        pdf = df[df['metric_name'] == i].copy()
        pdf = pdf.sort_values(by=['number'], ascending=True)
        value = list(pdf['value'])[:4]
        ratio_10_list.append(format(value[0], '.4f'))
        ratio_20_list.append(format(value[1], '.4f'))
        ratio_30_list.append(format(value[2], '.4f'))
        ratio_40_list.append(format(value[3], '.4f'))
    df_re['metric_name'] = metric_name_list
    df_re['10%'] = ratio_10_list
    df_re['20%'] = ratio_20_list
    df_re['30%'] = ratio_30_list
    df_re['40%'] = ratio_40_list
    df_re.to_excel('dog_result.xlsx', index=False)


if __name__ == '__main__':
    df_re = save_result()
    save_result()




