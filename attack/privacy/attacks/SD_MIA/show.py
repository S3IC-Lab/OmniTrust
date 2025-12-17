import numpy as np
import matplotlib.pyplot as plt


def distribution(member_scores, nonmember_scores, attack, filter, data):
    member_mean = np.mean(member_scores.numpy())
    nonmember_mean = np.mean(nonmember_scores.numpy())

    plt.figure(figsize=(10, 10))

    # 设置背景颜色为浅浅灰色
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')

    # Create histograms with white borders
    plt.hist(member_scores.numpy(), bins=50, color='#7AB656', alpha=0.5, label='Member Scores', edgecolor='white', linewidth=2, zorder=2)
    plt.hist(nonmember_scores.numpy(), bins=50, color='#DBB428', alpha=0.5, label='Hold-out Scores', edgecolor='white', linewidth=2, zorder=2)

    plt.xlabel('Scores', fontsize=44)
    plt.ylabel('Number of Samples', fontsize=44)

    plt.ylim(0, 40) # coco:180  # flickr:80 # pokemon:40
    # Bold the borders of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(4)  # Set border width

    # Mark the mean positions
    plt.axvline(member_mean, color='green', linestyle='dashed', linewidth=4, zorder=3)
    plt.axvline(nonmember_mean, color='orange', linestyle='dashed', linewidth=4, zorder=3)

    # 添加白色网格
    plt.grid(True, color='white', linestyle='-', linewidth=3, zorder=1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(fontsize=32, frameon=True, borderpad=1, loc='upper right')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    # save_path = './Distribution'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # filename = os.path.join(save_path, f"{data}-{attack}-{filter}.svg")
    # plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == '__main__':
    highs = np.random.rand(1000)
    scores = np.random.rand(1000)

    # 调用函数
    score_show(x=highs, y=scores)

