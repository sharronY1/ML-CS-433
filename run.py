"""读取、预处理并保存数据，最后调用全量训练和预测。"""

import model
import csv
import json
from pathlib import Path
import numpy as np
from helpers import load_csv_data


def load_data(data_path):
    """读取原始训练特征、标签和列名；清洗由统一转换函数执行。"""
    data_path = str(data_path)
    with open(Path(data_path) / "x_train.csv", encoding="utf-8-sig") as f:
        names = next(csv.reader(f))[1:]
    print(list(enumerate(names)))

    # 赋值给变量数据
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    return x_train, y_train, names

# 尝试初步清洗，总结了几条效果比较好的规则（特殊情况）：

    # 本次验证：5000 次迭代、学习率 0.01，五个原有候选阈值中的最佳 F1=0.42674160。
    # ===== 根据 BRFSS 2015 数据手册，把数据中那些实际上代表“缺失”的特殊数字转换成 np.nan =====
    # 因为有些缺失值并没有写成 NaN，而是用特殊数字表示
    # 这里只修改指定特征的值；原来的划分、填补、标准化和模型代码保持不变。
    # 来源：https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
def clean_columns(x, names):
    """按原有顺序执行清洗规则，训练集与测试集共用。"""
    x = np.array(x, dtype=float, copy=True)
    
    # 做一个字典记录特殊情况的列以及他缺失值的时候显示什么
    _missing_code_groups = {
        (
            3,
        ): """
        _AGE65YR
    """.split(),
        (
            7,
        ): """
        VIDFCLT2 VIREDIF3 VICTRCT4 VIGLUMA2 VIMACDG2 ASATTACK ASNOSLEP
    """.split(),
        (
            7,
            9,
        ): """
        LANDLINE GENHLTH HLTHPLN1 PERSDOC2 MEDCOST CHECKUP1 BPHIGH4 BPMEDS BLOODCHO CHOLCHK
        TOLDHI2 CVDSTRK3 ASTHMA3 ASTHNOW CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY
        DIABETE3 RENTHOM1 NUMHHOL2 NUMPHON2 CPDEMO1 VETERAN3 INTERNET PREGNANT QLACTLM2 USEEQUIP
        BLIND DECIDE DIFFWALK DIFFDRES DIFFALON SMOKE100 SMOKDAY2 STOPSMK2 USENOW3 EXERANY2
        LMTJOIN3 ARTHDIS2 ARTHSOCL SEATBELT FLUSHOT6 PNEUVAC3 HIVTST6 PDIABTST PREDIAB1 EYEEXAM
        DIABEYE DIABEDU CAREGIV1 CRGVLNG1 CRGVHRS1 CRGVPERS CRGVHOUS CRGVMST2 CRGVEXPT VIPRFVS2
        VIEYEXM2 VIINSUR2 CIMEMLOS CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS WTCHSALT DRADVISE
        ASYMPTOM ASTHMED3 ASINHALR HAREHAB1 STREHAB1 CVDASPRN ASPUNSAF RLIVPAIN RDUCHART
        RDUCSTRK ARTTODAY ARTHWGT ARTHEXER ARTHEDU TETANUS HPVADVC2 SHINGLE2 HADMAM HOWLONG
        HADPAP2 LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM LENGEXAM BLDSTOOL LSTBLDS3 HADSIGM3
        HADSGCO1 LASTSIG3 PCPSAAD2 PCPSADI1 PCPSARE1 PSATEST1 PSATIME PCPSARS1 PCDMDECN SCNTMNY1
        SCNTMEL1 SCNTPAID SCNTLPAD SXORIENT TRNSGNDR RCSRLTN2 CASTHDX2 CASTHNO2 EMTSUPRT
        LSATISFY MISTMNT ADANXEV DRNKANY5
    """.split(),
        (
            9,
        ): """
        MARITAL EDUCA EMPLOY1 INSULIN PCPSADE1 RCSGENDR _CHISPNC _RFHLTH _HCVU651 _RFHYPE5
        _CHOLCHK _RFCHOL _LTASTH1 _CASTHM1 _ASTHMS1 _HISPANC _RACE _RACEG21 _RACEGR3 _RFBMI5
        _CHLDCNT _EDUCAG _INCOMG _SMOKER3 _RFSMOK3 _RFBING5 _RFDRHV5 _FRTLT1 _VEGLT1 _TOTINDA
        PAMISS1_ _PACAT1 _PAINDX1 _PA150R2 _PA300R2 _PA30021 _PASTRNG _PAREC1 _PASTAE1 _LMTACT1
        _LMTWRK1 _LMTSCL1 _RFSEAT2 _RFSEAT3 _FLSHOT6 _PNEUMO2 _AIDTST3
    """.split(),
        (
            14,
        ): """
        _AGEG5YR
    """.split(),
        (
            77,
            99,
        ): """
        HHADULT PHYSHLTH MENTHLTH POORHLTH INCOME2 LASTSMK2 AVEDRNK2 DRNK3GE5 MAXDRNKS EXRACT11
        EXRACT21 JOINPAIN IMFVPLAC WHRTST10 DOCTDIAB CHKHEMO3 FEETCHK CRGVREL1 CRGVPRB1 VINOCRE2
        HPVADSHT ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE _CRACE1 _CPRACE
        _PRACE1 _MRACE1
    """.split(),
        (
            97,
            99,
        ): """
        SCNTWRK1 SCNTLWK1
    """.split(),
        (
            98,
        ): """
        ASERVIST ASDRVIST
    """.split(),
        (
            98,
            99,
        ): """
        DIABAGE2 ASTHMAGE ASRCHKUP
    """.split(),
        (
            99,
        ): """
        CHILDREN
    """.split(),
        (
            777,
            999,
        ): """
        ALCDAY5 FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1 EXEROFT1 EXERHMM1 EXEROFT2
        EXERHMM2 STRENGTH BLDSUGAR FEETCHK2 LONGWTCH ASACTLIM
    """.split(),
        (
            900,
        ): """
        DROCDY3_
    """.split(),
        (
            7777,
            9999,
        ): """
        WEIGHT2 HEIGHT3
    """.split(),
        (
            99000,
        ): """
        PAFREQ1_ PAFREQ2_ STRFREQ_
    """.split(),
        (
            99900,
        ): """
        _DRNKWEK MAXVO2_ FC60_
    """.split(),
        (
            99999,
        ): """
        WTKG3
    """.split(),
        (
            777777,
            999999,
        ): """
        FLSHTMY2 HIVTSTD3
    """.split(),
    }
    # 替换成nan
    for _codes, _columns in _missing_code_groups.items():
        for _name in _columns:
            _j = names.index(_name)
            for _x in (x,):
                _x[np.isin(_x[:, _j], _codes), _j] = np.nan

    # 以下派生列在课程 CSV 中已换算小数位，按实际单位处理特殊编码。
    # MAXVO2_、FC60_ 的 99900 已变为 999；三个活动频率列的 99000 已变为 99。
    for _name, _code in {
        "MAXVO2_": 999,
        "FC60_": 999,
        "PAFREQ1_": 99,
        "PAFREQ2_": 99,
        "STRFREQ_": 99,
    }.items():
        _j = names.index(_name)
        for _x in (x,):
            _x[_x[:, _j] == _code, _j] = np.nan

    # 把特殊编码变成 0，其含义真的是0
    _zero_code_groups = {
        (
            88,
        ): """
        PHYSHLTH MENTHLTH POORHLTH CHILDREN DRNK3GE5 DOCTDIAB CHKHEMO3 FEETCHK ASERVIST ASDRVIST
        ASRCHKUP ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE
    """.split(),
        (
            555,
        ): """
        FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1
    """.split(),
        (
            888,
        ): """
        ALCDAY5 STRENGTH BLDSUGAR FEETCHK2 ASACTLIM
    """.split(),
    }
    for _codes, _columns in _zero_code_groups.items():
        for _name in _columns:
            _j = names.index(_name)
            for _x in (x,):
                _x[np.isin(_x[:, _j], _codes), _j] = 0

    # 单位换算按官方字段定义；最后的 log1p 是本次验证过的实验变换。单位变换
    _feature_index = {name: j for j, name in enumerate(names)}
    for x in (x,):
        index = _feature_index
        # 把原始体重、身高的单位前缀解码成统一的千克、厘米。
        j = index["WEIGHT2"]
        a = x[:, j].copy()
        x[:, j] = np.nan
        m = (a >= 50) & (a <= 999)
        x[m, j] = a[m] * 0.45359237
        m = (a >= 9000) & (a <= 9998)
        x[m, j] = a[m] - 9000
        j = index["HEIGHT3"]
        a = x[:, j].copy()
        x[:, j] = np.nan
        m = (a >= 200) & (a <= 711) & (a % 100 < 12)
        x[m, j] = (a[m] // 100 * 12 + a[m] % 100) * 2.54
        m = (a >= 9000) & (a <= 9998)
        x[m, j] = a[m] - 9000
        # 周/月频率统一到每天；零频率保持为零。
        for name in ["ALCDAY5", "EXEROFT1", "EXEROFT2", "STRENGTH"]:
            j = index[name]
            a = x[:, j].copy()
            out = np.full(len(a), np.nan)
            out[a == 0] = 0
            m = (a >= 101) & (a <= 199)
            out[m] = (a[m] - 100) / 7
            m = (a >= 201) & (a <= 299)
            out[m] = (a[m] - 200) / 30
            x[:, j] = out
        for name in ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"]:
            j = index[name]
            a = x[:, j].copy()
            out = np.full(len(a), np.nan)
            out[a == 0] = 0
            m = (a >= 101) & (a <= 199)
            out[m] = a[m] - 100
            m = (a >= 201) & (a <= 299)
            out[m] = (a[m] - 200) / 7
            m = (a >= 301) & (a <= 399)
            out[m] = (a[m] - 300) / 30
            # 少于每月一次没有确切频率，不凭空指定一个数值，留待中位数填补。
            x[:, j] = out
        for name in ["EXERHMM1", "EXERHMM2"]:
            j = index[name]
            a = x[:, j].copy()
            x[:, j] = np.floor(a / 100) * 60 + a % 100
        for name in [
            "WEIGHT2",
            "WTKG3",
            "_BMI5",
            "ALCDAY5",
            "DROCDY3_",
            "_DRNKWEK",
            "EXEROFT1",
            "EXEROFT2",
            "STRENGTH",
            "EXERHMM1",
            "EXERHMM2",
        ]:
            j = index[name]
            x[:, j] = np.log1p(x[:, j])

    return x

#新增特征列，处理
def _process_features(raw, names, params=None, standardization_rows=None):
    """按原顺序执行唯一一套特征处理：拟合时计算统计，转换时复用统计。"""
    fitting = params is None
    x_tr = clean_columns(raw, names)
    # ===== 新增：记录清洗后、填补前的缺失位置 =====
    # 这247列由独立实验的训练部分筛出：缺失/非缺失均至少100次，并去除重复标记。
    # 固定已验证的列名单，避免改变特征顺序；这里不使用验证标签。
    _missing_feature_names = """
    CTELENUM STATERES NUMADULT NUMMEN CTELNUM1 CADULT CCLGHOUS
    CSTATE LANDLINE HHADULT GENHLTH PHYSHLTH MENTHLTH POORHLTH
    HLTHPLN1 PERSDOC2 MEDCOST CHECKUP1 BPHIGH4 BPMEDS BLOODCHO
    CHOLCHK TOLDHI2 CVDSTRK3 ASTHMA3 ASTHNOW CHCSCNCR CHCOCNCR
    CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY DIABETE3 DIABAGE2 MARITAL
    EDUCA RENTHOM1 NUMHHOL2 NUMPHON2 CPDEMO1 VETERAN3 EMPLOY1
    CHILDREN INCOME2 INTERNET WEIGHT2 HEIGHT3 PREGNANT QLACTLM2
    USEEQUIP BLIND DECIDE DIFFWALK DIFFDRES DIFFALON SMOKE100
    SMOKDAY2 STOPSMK2 LASTSMK2 USENOW3 ALCDAY5 AVEDRNK2 DRNK3GE5
    MAXDRNKS FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1
    EXERANY2 EXRACT11 EXEROFT1 EXERHMM1 EXRACT21 EXEROFT2 EXERHMM2
    STRENGTH LMTJOIN3 ARTHDIS2 ARTHSOCL JOINPAIN SEATBELT FLUSHOT6
    FLSHTMY2 IMFVPLAC PNEUVAC3 HIVTST6 HIVTSTD3 WHRTST10 PDIABTST
    PREDIAB1 INSULIN BLDSUGAR FEETCHK2 DOCTDIAB CHKHEMO3 FEETCHK
    EYEEXAM DIABEYE DIABEDU CAREGIV1 CRGVREL1 CRGVLNG1 CRGVHRS1
    CRGVPRB1 CRGVPERS CRGVHOUS CRGVMST2 CRGVEXPT VIDFCLT2 VIREDIF3
    VIPRFVS2 VINOCRE2 VIEYEXM2 VIINSUR2 VICTRCT4 VIGLUMA2 VIMACDG2
    CIMEMLOS CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS WTCHSALT
    LONGWTCH DRADVISE ASTHMAGE ASATTACK ASERVIST ASDRVIST ASRCHKUP
    ASACTLIM ASYMPTOM ASNOSLEP ASTHMED3 ASINHALR HAREHAB1 STREHAB1
    CVDASPRN ASPUNSAF RLIVPAIN RDUCHART RDUCSTRK ARTTODAY ARTHWGT
    ARTHEXER ARTHEDU TETANUS HPVADVC2 HPVADSHT SHINGLE2 HADMAM
    HOWLONG HADPAP2 LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM
    LENGEXAM BLDSTOOL LSTBLDS3 HADSIGM3 HADSGCO1 LASTSIG3 PCPSAAD2
    PCPSADI1 PCPSARE1 PSATEST1 PSATIME PCPSARS1 PCPSADE1 PCDMDECN
    SCNTMNY1 SCNTMEL1 SCNTPAID SCNTWRK1 SCNTLPAD SCNTLWK1 SXORIENT
    TRNSGNDR RCSGENDR RCSRLTN2 CASTHDX2 CASTHNO2 EMTSUPRT LSATISFY
    ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK
    ADMOVE MISTMNT ADANXEV MSCODE _CHISPNC _CRACE1 _CLLCPWT
    _DUALCOR _HCVU651 _CHOLCHK _CASTHM1 _PRACE1 _HISPANC _RACE
    _AGEG5YR HTIN4 HTM4 WTKG3 _BMI5 _SMOKER3 _RFBING5
    _DRNKWEK FTJUDA1_ FRUTDA1_ BEANDAY_ GRENDAY_ ORNGDAY_ VEGEDA1_
    _FRUTSUM _VEGESUM ACTIN11_ ACTIN21_ PAFREQ1_ PAFREQ2_ _MINAC11
    _MINAC21 STRFREQ_ PAMIN11_ PAMIN21_ PA1MIN_ PAVIG11_ PAVIG21_
    PA1VIGM_ _PACAT1 _PAINDX1 _PAREC1 _LMTACT1 _LMTWRK1 _LMTSCL1
    _FLSHOT6 _PNEUMO2
""".split()
    if not fitting:
        _missing_feature_names = np.asarray(params["missing_names"]).tolist()
    _missing_feature_indices = [names.index(_name) for _name in _missing_feature_names]
    missing_tr = np.isnan(x_tr[:, _missing_feature_indices]).astype(float)
    # ===== 缺失位置记录结束 =====

    # 算中位数填补缺失值
    medians = np.nanmedian(x_tr, axis=0) if fitting else params["medians"]
    # 特殊情况一列全是nan，设为0
    medians = np.where(np.isnan(medians), 0, medians)
    # 补充进去
    x_tr = np.where(np.isnan(x_tr), medians, x_tr)

    # ===== 新增：两列类别特征的独热编码 =====
    # 在标准化之前使用原编码；类别3为参照，只添加类别1、2、4的0/1列。
    # 未知/拒答已转为缺失并按原规则填补，不额外改变缺失值的填补方式。
    _onehot_feature_names = ["DIABETE3", "BPHIGH4"]
    if not fitting:
        _onehot_feature_names = np.asarray(params["onehot_names"]).tolist()
    _onehot_feature_indices = [names.index(_name) for _name in _onehot_feature_names]
    _onehot_tr = np.column_stack(
        [
            x_tr[:, _j] == _category
            for _j in _onehot_feature_indices
            for _category in [1, 2, 4]
        ]
    ).astype(float)
    # ===== 独热编码准备结束 =====

    # 标准化
    means = np.mean(x_tr, axis=0) if fitting else params["means"]
    stds = np.std(x_tr, axis=0) if fitting else params["stds"]
    # 防止某一列一样全是0
    stds = np.where(stds == 0, 1, stds)
    x_tr = (x_tr - means) / stds

    # 新增交互方案：health_pairs；5000 次迭代，gamma=0.01。
    # 当前验证集：总体 F1=0.42738266，40 岁以下 F1=0.17297297。
    # ===== 新增：经验证的交互与非线性特征，保留所有原有特征 =====
    # 在原有标准化后生成；新增列的均值和标准差只用训练部分计算。

    # 交互项的计算与参数导出共用同一份定义，保持列顺序一致。
    _extra_feature_spec = {
        "pairs": [
            ["GENHLTH", "PHYSHLTH"],
            ["GENHLTH", "DIFFWALK"],
            ["PHYSHLTH", "DIFFWALK"],
            ["GENHLTH", "CVDSTRK3"],
            ["GENHLTH", "DIABETE3"],
            ["_BMI5", "DIFFWALK"],
            ["BPHIGH4", "DIABETE3"],
            ["CVDSTRK3", "DIFFWALK"],
        ],
        "squares": ["GENHLTH", "PHYSHLTH", "_BMI5"],
    }

    if not fitting:
        _extra_feature_spec = json.loads(str(params["extra_spec_json"]))

    # 新增空列
    _extra_tr = []
    # 健康指标之间的交互项，两个原有主特征仍然保留。一共是8对新的，假设其中的两个特征会互相影响
    for _left, _right in _extra_feature_spec["pairs"]:
        _l = names.index(_left)
        _r = names.index(_right)
        _extra_tr.append(x_tr[:, _l] * x_tr[:, _r])
    # 平方项允许单个指标的作用随取值发生弯曲。
    for _name in _extra_feature_spec["squares"]:
        _j = names.index(_name)
        _extra_tr.append(x_tr[:, _j] ** 2)
    # 拼接成新矩阵
    _extra_tr = np.column_stack(_extra_tr)
    _extra_means = np.mean(_extra_tr, axis=0) if fitting else params["extra_means"]
    _extra_stds = np.std(_extra_tr, axis=0) if fitting else params["extra_stds"]
    _extra_stds = np.where(_extra_stds == 0, 1, _extra_stds)
    x_tr = np.column_stack([x_tr, (_extra_tr - _extra_means) / _extra_stds])
    # 至此新增十一列
    # ===== 年龄交互特征结束 =====

    # ===== 新增：采用两列独热编码＋清洗后缺失标记 =====
    # 必须等原来的健康交互项全部生成后，才替换两列数值主效应，防止names索引错位。
    # 原来的交互项保留；基础处理保留0/1编码；额外标准化由model.py统一完成。
    _keep_numeric_columns = np.ones(x_tr.shape[1], dtype=bool)
    # 其余列全部保留，唯独删除原来的 DIABETE3 和 BPHIGH4 两列
    _keep_numeric_columns[_onehot_feature_indices] = False
    x_tr = np.column_stack([x_tr[:, _keep_numeric_columns], _onehot_tr, missing_tr])
    # onehot一共两列，增添变成了六列，把之前的两列删了
    # 321原始+11交互平方项-2原类别列+6onehot+247缺失标记+1截距=584列
    # ===== 新方案拼接结束 =====

    # 加截距
    x_tr = np.column_stack([np.ones(len(x_tr)), x_tr])

    # 最后统一标准化；全量时使用全部行，验证时只使用实际拟合行。
    _fit_x = x_tr if standardization_rows is None else x_tr[standardization_rows]
    final_means = _fit_x.mean(axis=0) if fitting else params["final_means"]
    final_stds = _fit_x.std(axis=0) if fitting else params["final_stds"]
    final_means[0] = 0.0
    final_stds[0] = 1.0
    final_stds = np.where(final_stds == 0, 1.0, final_stds)
    del _fit_x
    x_tr -= final_means
    x_tr /= final_stds

    if fitting:
        params = dict(
            medians=medians, means=means, stds=stds,
            extra_means=_extra_means, extra_stds=_extra_stds,
            final_means=final_means, final_stds=final_stds,
            names=np.asarray(names),
            extra_spec_json=json.dumps(_extra_feature_spec, ensure_ascii=False),
            missing_names=np.asarray(_missing_feature_names),
            onehot_names=np.asarray(_onehot_feature_names),
        )
    return x_tr, params

#不传已有的 params，让 _process_features() 根据训练数据计算预处理参数并且返回处理好的训练数据
def fit_preprocessing(x, names, standardization_rows=None):
    """计算预处理参数，同时返回处理好的训练数据。"""
    return _process_features(x, names, standardization_rows=standardization_rows)

def prepare_features(
    x_tr,
    y_tr,
    names,
    cache_path,
    model_dir,
    data_path,
    x_val=None,
    y_val=None,
    standardization_rows=None,
):
    """处理传入的数据并保存，不进行训练/验证划分。"""
    data_path = str(data_path)
    cache_path = Path(cache_path)
    model_dir = Path(model_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if x_val is None:
        x_val = np.empty((0, x_tr.shape[1]))
    if y_val is None:
        y_val = np.empty(0)

    x_tr, params = fit_preprocessing(x_tr, names, standardization_rows)
    x_val, _ = _process_features(x_val, names, params=params)

    # 转换成01，标签一开始是-1，1
    y_tr_01 = (y_tr + 1) / 2
    y_val_01 = (y_val + 1) / 2

    # 把处理的部分保存
    np.savez(
        cache_path,
        x_tr=x_tr,
        y_tr_01=y_tr_01,
        x_val=x_val,
        y_val_01=y_val_01,
    )

    # ===== 新增：保存测试集预测需要的预处理参数，不训练模型 =====
    # 列数是否正确
    if x_tr.shape[1] != 584:
        raise ValueError("特征数量不正确")
    # 保存预处理时的参数
    _preprocessing_path = model_dir / "preprocessing.npz"
    np.savez(
        _preprocessing_path,
        **params,
    )
    print("已导出预处理参数：", _preprocessing_path)
    # ===== 预处理参数导出结束 =====


    return dict(
        cache_path=str(cache_path),
        model_dir=str(model_dir),
        training_rows=len(y_tr),
        validation_rows=len(y_val),
    )


def prepare_data(data_path, project_dir=None):
    """读取全部训练数据，清洗、处理并保存。"""
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent
    project_dir = Path(project_dir)
    x_train, y_train, names = load_data(data_path)
    return prepare_features(
        x_train, y_train, names,
        project_dir / "cache/prepared_full_data.npz",
        project_dir / "models", data_path,
    )


def main():
    import predict_submission

    project = Path(__file__).resolve().parent
    data_path = project / "data/dataset_to_release/dataset"
    output = project / "results/submission.csv"
    prepare_data(data_path, project_dir=project)
    model.train_full(project)
    predict_submission.generate_submission(data_path, output, project)


if __name__ == "__main__":
    main()
