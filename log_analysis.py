import re, os
LOG_FOLD = "logs/DGAP"
LOG_PREFIX = "edm_qm9_DGAP"

def get_log_files():
    '''
    there may be multiple log files, we need to read them all and combine them into one list
    use regex to extract the log files
    '''
    # 遍历log文件夹，读取所有和LOG_PREFIX匹配的log文件
    log_files = [os.path.join(LOG_FOLD, f) for f in os.listdir(LOG_FOLD) if re.match(f"{LOG_PREFIX}.*\.log", f)]
    return log_files

def analysis_pred_rate(log_files):
    '''
    demo: Epoch: 66, iter: 986/1563, Loss 2.72, NLL: 2.72, RegTerm: 0.0, GradNorm: 4.6, denoise x: 0.176 , pred_loss: 0.052, pred_rate: 1.000
    extract the pred_rate from each line of the log file
    '''
    pred_rate_list = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if "pred_rate" in line:
                    pred_rate = float(line.split()[-1])
                    #需要去除逗号
                    epoch = int(line.split()[1].split(",")[0])
                    pred_rate_list.append((epoch, pred_rate))
    pred_rate_list.sort(key=lambda x: x[0])
    print("Mean pred_rate:", sum([x[1] for x in pred_rate_list]) / len(pred_rate_list))

# 绘制变化曲线
def validity_uniqueness_novelty_analysis(log_files):
    '''
    Validity 0.5850, Uniqueness: 1.0000, Novelty: 0.9829

    '''
    validity_list = []
    uniqueness_list = []
    novelty_list = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if "Validity" in line and "Uniqueness" in line and "Novelty" in line:
                    validity = float(line.split()[1].split(",")[0])
                    uniqueness = float(line.split()[3].split(",")[0])
                    novelty = float(line.split()[5].split(",")[0])
                    print(f"Validity: {validity}, Uniqueness: {uniqueness}, Novelty: {novelty}")
                    validity_list.append(validity)
                    uniqueness_list.append(uniqueness)
                    novelty_list.append(novelty)
    import matplotlib.pyplot as plt
    plt.plot(validity_list, label="Validity")
    plt.plot(uniqueness_list, label="Uniqueness")
    plt.plot(novelty_list, label="Novelty")
    plt.legend()
    plt.show()
                    

def main():
    log_files = get_log_files()
    # analysis_pred_rate(log_files)
    validity_uniqueness_novelty_analysis(log_files)
    pass

if __name__ == '__main__':
    main()