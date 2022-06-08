import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mplfonts import use_font

use_font('Noto Serif CJK SC')#指定中文字体

plt.rcParams['savefig.dpi'] = 1200 #图片像素
plt.rcParams['figure.dpi'] = 1200 #分辨率
#
# x = np.array([0.735, 0.749, 0.725, 0.698, 0.743, 0.658, 0.686, 0.753, 0.706, 0.772])
# y = np.array([0.721, 0.733, 0.729, 0.708, 0.752, 0.657, 0.666, 0.730, 0.690, 0.760])
#
# # plt.title("SCATTER PLOT")
# plt.xlabel("模型真实性能（mAP）",fontsize='15')
# plt.ylabel("基于超网评估的模型性能（mAP）",fontsize='15')
# plt.scatter(x, y)
#
# plt.plot([0.65, 0.78],[0.65, 0.78],ls='--',c='red',linewidth='0.5')
# # label = ['t', 't**2']
# # plt.legend(label, loc='upper left')
# plt.savefig('./test4.jpg', dpi=1200)
# plt.show()

# map=[(0.7029961347579956,), (0.7769811749458313,), (0.7499879002571106,), (0.7782742977142334,), (0.7276335954666138,), (0.7690093517303467,), (0.7711813449859619,), (0.760474443435669,), (0.7923439145088196,), (0.7300735712051392,), (0.6712273359298706,), (0.7841442227363586,), (0.712496280670166,), (0.7919647097587585,), (0.7052847146987915,), (0.7858890295028687,), (0.7573205232620239,), (0.788731575012207,), (0.776030421257019,), (0.7686319351196289,), (0.7439572215080261,), (0.7312048673629761,), (0.7813869714736938,), (0.7084263563156128,), (0.7177879810333252,), (0.7265358567237854,), (0.7026268243789673,), (0.7837623357772827,), (0.7682516574859619,), (0.681511402130127,), (0.7607131004333496,), (0.6886653900146484,), (0.7356616854667664,), (0.7602477669715881,), (0.7590325474739075,), (0.7281590700149536,), (0.7531466484069824,), (0.7687210440635681,), (0.7147676944732666,), (0.7311006784439087,), (0.7633618116378784,), (0.7439562678337097,), (0.7100242376327515,), (0.7351633906364441,), (0.7624781727790833,), (0.7464994192123413,), (0.7612746953964233,), (0.7530050277709961,), (0.727807343006134,), (0.7558906078338623,), (0.7224870324134827,), (0.6819302439689636,), (0.6536002159118652,), (0.7573886513710022,), (0.7354097962379456,), (0.7760394215583801,), (0.6889123320579529,), (0.673540472984314,), (0.7839279770851135,), (0.7477871179580688,), (0.7690457701683044,), (0.7253055572509766,), (0.699880838394165,), (0.7636493444442749,), (0.7222591042518616,), (0.7154511213302612,), (0.6973277926445007,), (0.7294648289680481,), (0.7722289562225342,), (0.7484341859817505,), (0.7179838418960571,), (0.7789587378501892,), (0.7661852836608887,), (0.6740143299102783,), (0.7739219069480896,), (0.7522213459014893,), (0.7349086999893188,), (0.7474586367607117,), (0.7451187372207642,), (0.7284987568855286,), (0.7286466956138611,), (0.7428269982337952,), (0.7269575595855713,), (0.7764610052108765,), (0.7227656841278076,), (0.7297836542129517,), (0.7675607204437256,), (0.7454067468643188,), (0.7923897504806519,), (0.7749951481819153,), (0.6650556325912476,), (0.6993907690048218,), (0.7248772978782654,), (0.744205892086029,), (0.7481426000595093,), (0.766386866569519,), (0.6688883304595947,), (0.7531068325042725,), (0.7439278364181519,), (0.7254022359848022,), (0.7415995001792908,), (0.7141093015670776,), (0.783927321434021,), (0.7602688670158386,), (0.7020801901817322,), (0.7590440511703491,), (0.6713242530822754,), (0.7341712713241577,), (0.7405189275741577,), (0.7056962251663208,), (0.6473800539970398,), (0.7290037274360657,), (0.6929726600646973,), (0.6797917485237122,), (0.7227057218551636,), (0.7828299403190613,), (0.7336580753326416,), (0.7667636871337891,), (0.7428551316261292,), (0.7604559659957886,), (0.7380090951919556,), (0.7466524243354797,), (0.744218647480011,), (0.7399031519889832,), (0.6620960831642151,), (0.7361273169517517,), (0.6798279285430908,), (0.7391996383666992,), (0.7452713251113892,), (0.7455366849899292,), (0.6654543280601501,), (0.7305737137794495,), (0.7089305520057678,), (0.7230896949768066,), (0.7764816284179688,), (0.7550553679466248,), (0.7327175736427307,), (0.7171229124069214,), (0.7508266568183899,), (0.7392698526382446,), (0.6611971855163574,), (0.7724143862724304,), (0.7242786288261414,), (0.7466968297958374,), (0.7441290616989136,), (0.7388807535171509,), (0.71357262134552,), (0.7760267853736877,), (0.6921421885490417,), (0.7340751886367798,), (0.7787094712257385,), (0.7747469544410706,), (0.7595776915550232,), (0.7325901389122009,), (0.6598588228225708,), (0.6991742849349976,), (0.7523238062858582,), (0.6736222505569458,), (0.6847564578056335,), (0.7802672982215881,), (0.7576698064804077,), (0.760796070098877,), (0.7066744565963745,), (0.7641741633415222,), (0.7737361192703247,), (0.6726661920547485,), (0.7242861986160278,), (0.7471008896827698,), (0.7096966505050659,), (0.7182643413543701,), (0.7258776426315308,), (0.7511224746704102,), (0.7485682964324951,), (0.7570759057998657,), (0.7248622179031372,), (0.7321769595146179,), (0.7635930180549622,), (0.7552860975265503,), (0.7408198118209839,), (0.7272698283195496,), (0.7323681116104126,), (0.7387614846229553,), (0.7418978810310364,), (0.705730140209198,), (0.7237622737884521,), (0.7235435247421265,), (0.6980141401290894,), (0.7282820343971252,), (0.7529776096343994,), (0.7445058822631836,), (0.7145021557807922,), (0.7233415842056274,), (0.6688505411148071,), (0.7029539346694946,), (0.7725902795791626,), (0.7072466611862183,), (0.7339847087860107,), (0.7803710103034973,), (0.6799358129501343,), (0.7483887672424316,)]
# mAP=[]
# for item in map:
#     mAP.append(item[0])
mAP=[0.7029961347579956, 0.7769811749458313, 0.7499879002571106, 0.7782742977142334, 0.7276335954666138, 0.7690093517303467, 0.7711813449859619, 0.760474443435669, 0.7923439145088196, 0.7300735712051392, 0.6712273359298706, 0.7841442227363586, 0.712496280670166, 0.7919647097587585, 0.7052847146987915, 0.7858890295028687, 0.7573205232620239, 0.788731575012207, 0.776030421257019, 0.7686319351196289, 0.7439572215080261, 0.7312048673629761, 0.7813869714736938, 0.7084263563156128, 0.7177879810333252, 0.7265358567237854, 0.7026268243789673, 0.7837623357772827, 0.7682516574859619, 0.681511402130127, 0.7607131004333496, 0.6886653900146484, 0.7356616854667664, 0.7602477669715881, 0.7590325474739075, 0.7281590700149536, 0.7531466484069824, 0.7687210440635681, 0.7147676944732666, 0.7311006784439087, 0.7633618116378784, 0.7439562678337097, 0.7100242376327515, 0.7351633906364441, 0.7624781727790833, 0.7464994192123413, 0.7612746953964233, 0.7530050277709961, 0.727807343006134, 0.7558906078338623, 0.7224870324134827, 0.6819302439689636, 0.6536002159118652, 0.7573886513710022, 0.7354097962379456, 0.7760394215583801, 0.6889123320579529, 0.673540472984314, 0.7839279770851135, 0.7477871179580688, 0.7690457701683044, 0.7253055572509766, 0.699880838394165, 0.7636493444442749, 0.7222591042518616, 0.7154511213302612, 0.6973277926445007, 0.7294648289680481, 0.7722289562225342, 0.7484341859817505, 0.7179838418960571, 0.7789587378501892, 0.7661852836608887, 0.6740143299102783, 0.7739219069480896, 0.7522213459014893, 0.7349086999893188, 0.7474586367607117, 0.7451187372207642, 0.7284987568855286, 0.7286466956138611, 0.7428269982337952, 0.7269575595855713, 0.7764610052108765, 0.7227656841278076, 0.7297836542129517, 0.7675607204437256, 0.7454067468643188, 0.7923897504806519, 0.7749951481819153, 0.6650556325912476, 0.6993907690048218, 0.7248772978782654, 0.744205892086029, 0.7481426000595093, 0.766386866569519, 0.6688883304595947, 0.7531068325042725, 0.7439278364181519, 0.7254022359848022, 0.7415995001792908, 0.7141093015670776, 0.783927321434021, 0.7602688670158386, 0.7020801901817322, 0.7590440511703491, 0.6713242530822754, 0.7341712713241577, 0.7405189275741577, 0.7056962251663208, 0.6473800539970398, 0.7290037274360657, 0.6929726600646973, 0.6797917485237122, 0.7227057218551636, 0.7828299403190613, 0.7336580753326416, 0.7667636871337891, 0.7428551316261292, 0.7604559659957886, 0.7380090951919556, 0.7466524243354797, 0.744218647480011, 0.7399031519889832, 0.6620960831642151, 0.7361273169517517, 0.6798279285430908, 0.7391996383666992, 0.7452713251113892, 0.7455366849899292, 0.6654543280601501, 0.7305737137794495, 0.7089305520057678, 0.7230896949768066, 0.7764816284179688, 0.7550553679466248, 0.7327175736427307, 0.7171229124069214, 0.7508266568183899, 0.7392698526382446, 0.6611971855163574, 0.7724143862724304, 0.7242786288261414, 0.7466968297958374, 0.7441290616989136, 0.7388807535171509, 0.71357262134552, 0.7760267853736877, 0.6921421885490417, 0.7340751886367798, 0.7787094712257385, 0.7747469544410706, 0.7595776915550232, 0.7325901389122009, 0.6598588228225708, 0.6991742849349976, 0.7523238062858582, 0.6736222505569458, 0.6847564578056335, 0.7802672982215881, 0.7576698064804077, 0.760796070098877, 0.7066744565963745, 0.7641741633415222, 0.7737361192703247, 0.6726661920547485, 0.7242861986160278, 0.7471008896827698, 0.7096966505050659, 0.7182643413543701, 0.7258776426315308, 0.7511224746704102, 0.7485682964324951, 0.7570759057998657, 0.7248622179031372, 0.7321769595146179, 0.7635930180549622, 0.7552860975265503, 0.7408198118209839, 0.7272698283195496, 0.7323681116104126, 0.7387614846229553, 0.7418978810310364, 0.705730140209198, 0.7237622737884521, 0.7235435247421265, 0.6980141401290894, 0.7282820343971252, 0.7529776096343994, 0.7445058822631836, 0.7145021557807922, 0.7233415842056274, 0.6688505411148071, 0.7029539346694946, 0.7725902795791626, 0.7072466611862183, 0.7339847087860107, 0.7803710103034973, 0.6799358129501343, 0.7483887672424316]
FLOPs=[11.43, 17.35, 13.88, 14.81, 11.01, 15.04, 16.07, 14.59, 19.26, 11.21, 9.61, 17.3, 11.58, 17.83, 11.09, 17.31, 13.85, 20.56, 18.01, 14.06, 15.04, 13.0, 14.73, 11.83, 10.52, 13.27, 11.29, 15.07, 14.24, 10.54, 10.81, 8.69, 11.84, 15.02, 13.9, 12.68, 16.1, 13.54, 9.59, 15.24, 15.83, 10.93, 11.3, 9.35, 16.04, 15.13, 15.16, 14.17, 11.99, 14.16, 10.29, 10.84, 6.34, 11.49, 12.88, 13.77, 7.01, 11.08, 16.96, 15.24, 16.49, 10.51, 12.45, 13.3, 11.67, 12.26, 12.22, 11.3, 16.93, 13.79, 10.05, 17.72, 15.97, 9.13, 18.05, 15.26, 12.91, 9.78, 12.16, 13.41, 10.7, 12.05, 12.42, 17.45, 9.54, 15.19, 15.44, 15.36, 19.33, 16.67, 7.88, 9.04, 11.87, 15.13, 14.91, 13.36, 10.14, 12.9, 12.46, 14.65, 16.55, 13.52, 17.62, 12.74, 11.22, 13.42, 9.55, 13.44, 11.58, 10.63, 6.26, 10.84, 7.93, 9.5, 12.07, 15.36, 13.87, 14.74, 11.05, 15.14, 9.48, 13.17, 11.71, 14.85, 8.78, 11.36, 9.19, 12.64, 12.6, 17.25, 10.74, 11.03, 9.88, 10.59, 18.04, 17.91, 12.97, 10.75, 16.23, 12.63, 8.35, 14.17, 11.81, 13.22, 14.58, 13.42, 11.13, 15.63, 11.71, 11.17, 17.46, 14.86, 14.29, 9.63, 8.45, 9.02, 15.76, 9.2, 8.31, 16.1, 14.17, 15.13, 12.44, 14.6, 14.12, 9.62, 14.31, 13.27, 13.17, 12.26, 12.56, 16.75, 15.08, 13.51, 14.19, 13.81, 13.32, 14.0, 12.66, 12.46, 14.32, 10.94, 12.06, 10.56, 12.26, 11.94, 11.86, 12.72, 13.51, 11.0, 12.28, 12.9, 7.91, 9.8, 17.17, 9.41, 11.98, 17.37, 11.34, 10.99]
Params=[3.26, 4.83, 4.25, 4.6, 2.96, 3.76, 5.76, 3.58, 6.23, 3.6, 2.69, 5.59, 3.03, 5.37, 2.21, 5.75, 4.79, 5.72, 4.2, 4.69, 3.79, 2.89, 5.08, 3.56, 4.01, 3.42, 3.71, 5.56, 4.78, 3.07, 4.21, 1.3, 3.37, 6.75, 3.52, 3.94, 3.38, 4.07, 1.88, 4.1, 3.34, 3.38, 2.05, 3.32, 5.36, 3.7, 4.51, 3.3, 3.27, 4.01, 3.84, 2.22, 1.61, 4.71, 3.71, 4.86, 2.35, 1.78, 5.7, 4.48, 3.72, 2.98, 2.24, 4.98, 2.49, 3.46, 3.55, 2.84, 4.97, 4.86, 3.65, 4.4, 4.29, 1.56, 5.19, 4.61, 3.01, 4.29, 3.69, 3.86, 4.76, 3.31, 2.68, 5.11, 4.28, 3.62, 5.04, 3.33, 5.95, 4.55, 1.36, 2.44, 2.74, 3.76, 3.36, 5.15, 1.75, 3.86, 3.67, 2.22, 3.45, 2.77, 5.42, 3.87, 3.02, 4.76, 1.87, 4.79, 3.41, 2.39, 1.28, 3.38, 2.29, 2.4, 2.89, 5.57, 3.11, 3.94, 4.34, 5.33, 4.36, 4.58, 3.94, 3.07, 1.64, 3.97, 1.58, 4.6, 3.9, 3.29, 2.66, 3.11, 2.16, 2.81, 4.05, 3.67, 3.08, 3.35, 3.53, 4.17, 1.87, 4.9, 2.66, 2.93, 3.69, 3.0, 3.03, 5.08, 2.51, 2.95, 6.47, 5.07, 5.78, 3.17, 2.51, 2.06, 3.75, 1.88, 1.94, 5.04, 4.25, 5.11, 2.6, 4.53, 5.97, 1.6, 3.76, 4.07, 3.43, 2.91, 2.76, 4.91, 4.15, 4.03, 2.89, 3.65, 5.39, 5.38, 3.24, 3.26, 2.93, 4.07, 3.88, 3.66, 2.27, 2.06, 3.46, 3.71, 3.1, 4.29, 2.11, 3.9, 1.45, 2.09, 3.6, 2.1, 3.72, 4.73, 2.4, 4.57]
# plt.xlabel("模型参数量（M）",fontsize='15')
plt.xlabel("模型计算量FLOPs（G）",fontsize='15')
plt.ylabel("基于超网评估的模型性能（mAP）",fontsize='15')
# plt.scatter(Params, mAP, s=10)
plt.scatter(FLOPs, mAP, s=10)
#
# plt.plot([0.65, 0.78],[0.65, 0.78],ls='--',c='red',linewidth='0.5')
# # label = ['t', 't**2']
# # plt.legend(label, loc='upper left')
plt.savefig('./test41.jpg', dpi=1200)