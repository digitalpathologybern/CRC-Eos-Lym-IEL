import pandas as pd
import numpy as np

# pd.set_option("future.no_silent_downcasting", True)


def get_data(split_normal=False):
    berger = pd.read_csv("raw_results/w_clinical/berger_w_clincal.csv")
    tcga = pd.read_csv("raw_results/w_clinical/tcga_w_clinical.csv")
    radboud = pd.read_csv(
        "raw_results/w_clinical/radboud_w_clinical.csv",
    )
    radboud["G"] = radboud["G"].replace({1.0: 0.0, 2.0: 0.0, 3.0: 1.0, 4.0: 1.0})
    toronto = pd.read_csv("raw_results/w_clinical/toronto_w_clinical.csv")
    bern = pd.read_csv("raw_results/w_clinical/bern1516_w_clinical.csv")

    berger = berger[berger.tnm.isin([2.0, 3.0])].copy()
    bern = bern[~bern.tnm.isin([0.0])].copy()

    radboud_exclude = pd.read_csv(
        "/storage/research/pathology_tru/Elias/other_ds/radboud/exclude_these.txt",
        header=None,
        sep=",",
    )[0]
    radboud = radboud[~radboud.patient_id.isin(radboud_exclude)].copy()
    toronto_exclude = pd.read_csv(
        "/storage/research/pathology_tru/Elias/other_ds/toronto/exclude_these.txt",
        header=None,
        sep=",",
    )[0]
    # toronto.loc[toronto.slide_id.isin(toronto_exclude)|toronto.patient_id.isin(toronto_exclude),["slide_id","patient_id"]]
    toronto = toronto[
        ~(
            toronto.slide_id.isin(toronto_exclude)
            | toronto.patient_id.isin(toronto_exclude)
        )
    ].copy()
    berger_exclude = pd.read_csv(
        "/storage/research/pathology_tru/Elias/other_ds/Berger_2022/Berger_2022/exclude_these.txt",
        header=None,
        sep=",",
    )[0]
    berger = berger[~berger.slide_id.str.contains("|".join(berger_exclude))].copy()

    radboud_immunoscore = pd.read_excel(
        "/storage/research/pathology_tru/Elias/other_ds/radboud/Immunoscore data to Bern.xlsx"
    )
    comp_bern = radboud_immunoscore[
        radboud_immunoscore["patient_ID"].astype(str).str.startswith("BER")
    ].copy()
    radboud_immunoscore = radboud_immunoscore[
        ~radboud_immunoscore["patient_ID"].astype(str).str.startswith("BER")
    ].copy()
    radboud_immunoscore["patient_ID"] = "P" + radboud_immunoscore.patient_ID.astype(
        str
    ).str.zfill(6)
    radboud_immunoscore["immunoscore"] *= 100
    radboud = radboud.merge(
        radboud_immunoscore[
            [
                "patient_ID",
                "CD3CT-mean-QC",
                "CD3IM-mean-QC",
                "CD8CT-mean-QC",
                "CD8IM-mean-QC",
                "immunoscore",
                "immunoscore_2_tier",
                "immunoscore_3_tier",
            ]
        ],
        left_on="patient_id",
        right_on="patient_ID",
        how="left",
    ).drop("patient_ID", axis=1)

    bern_immunoscore = pd.read_excel(
        "/storage/research/pathology_tru/Elias/other_ds/TMA_Bern_meta/bern_immunoscore.xlsx"
    )
    bern_immunoscore["NAME"] = bern_immunoscore.NAME.str.replace("BER", "").astype(int)
    bern_immunoscore.columns = [
        "patient_id",
        "immunoscore_2_tier",
        "immunoscore_3_tier",
        "immunoscore_5_tier",
        "immunoscore",
    ]
    bern_immunoscore["immunoscore_2_tier"] = (
        bern_immunoscore.immunoscore_2_tier.replace({"25-100%": 1, "0-25%": 0}).astype(
            float
        )
    )
    bern_immunoscore["immunoscore_3_tier"] = (
        bern_immunoscore.immunoscore_3_tier.replace(
            {"70-100%": 2, "0-25%": 0, "25-70%": 1}
        ).astype(float)
    )
    bern_immunoscore["immunoscore_5_tier"] = (
        bern_immunoscore.immunoscore_5_tier.replace(
            {"0-10%": 0, "10-25%": 1, "25-70%": 2, "70-95%": 3, "95-100%": 4}
        ).astype(float)
    )
    bern = bern.merge(bern_immunoscore, on="patient_id", how="left").copy()
    berger = berger.merge(bern_immunoscore, on="patient_id", how="left").copy()
    # combine
    comb = pd.concat([berger, bern, tcga, radboud, toronto])
    comb.gender = comb.gender.replace({"Female": 1.0, "Male": 0.0}).astype(float)
    comb.pT = comb.pT.replace(
        {
            "T3": 3.0,
            "T2": 2.0,
            "T4A": 4.0,
            "T4B": 4.0,
            "T1": 1.0,
            "TIS": np.NaN,
            "x": np.NaN,
            "T4": 4.0,
        }
    ).astype(float)
    comb.pN = comb.pN.replace(
        {
            "N0": 0.0,
            "N1": 1.0,
            "N2": 2.0,
            "N1B": 1.0,
            "N2B": 2.0,
            "N1A": 1,
            "N2A": 2.0,
            "N1C": 1,
            "x": np.NaN,
            "NX": np.NaN,
        }
    ).astype(float)
    comb.pM = comb.pM.replace(
        {"M0": 0, "MX": np.NaN, "M1": 1, "1a": 1, "M1A": 1, "M1B": 1, "1c": 1, "1b": 1}
    ).astype(float)
    comb.tnm = comb.tnm.replace(
        {
            "II": 2,
            "III": 3,
            "IV": 4,
            "I": 1,
            "2A": 2,
            "2B": 2,
            "2C": 2,
            "3A": 3,
            "3B": 3,
            "3C": 3,
            "4A": 4,
            "4B": 4,
            "4C": 4,
        }
    ).astype(float)
    comb = comb.reset_index().drop("index", axis=1)
    comb.location = comb.location.str.lower()

    comb.neo_adj = comb.neo_adj.replace({False: 0, True: 1}).astype(float)
    comb = comb[comb.neo_adj != 1]

    comb.location = comb.location.replace(
        {"right": 0.0, "left": 1.0, "rectum": 2.0}
    ).astype(float)
    bern_add_stats = pd.read_csv(
        "/storage/research/pathology_tru/Elias/other_ds/TMA_Bern_meta/new_master_file.csv"
    )[["ID", "Klintrup_Makinen"]]
    bern_add_stats["ID"] = bern_add_stats["ID"].astype(object)
    comb = comb.merge(bern_add_stats, left_on="patient_id", right_on="ID", how="left")

    tcga_stats = pd.read_excel(
        "/storage/research/pathology_tru/Elias/tcga/bilal_complete_data.xlsx"
    )[
        [
            "PATIENT",
            "ColorectalCMS",
            "Hypermutated",
            "ImmuneSubtype",
            "CIN_F_B_classification",
            "HypermethylationCategory",
            "KRAS_mutation",
            "BRAF_mutation",
        ]
    ]
    comb = comb.merge(
        tcga_stats, left_on="patient_id", right_on="PATIENT", how="left"
    )  #
    comb.ColorectalCMS = comb.ColorectalCMS.replace(
        {"CMS1": 1, "CMS2": 2, "CMS3": 3, "CMS4": 4}
    ).astype(float)
    comb.ImmuneSubtype = comb.ImmuneSubtype.replace(
        {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6}
    ).astype(float)
    comb.HypermethylationCategory = comb.HypermethylationCategory.replace(
        {"Non-CIMP": 0, "CIMP-H": 2, "CRC CIMP-L": 1, "GEA CIMP-L": 1}
    ).astype(float)
    comb.KRAS_mutation = comb.KRAS_mutation.replace({"MUT": 1, "WT": 0}).astype(float)
    comb.BRAF_mutation = comb.BRAF_mutation.replace({"MUT": 1, "WT": 0}).astype(float)

    if split_normal:
        comb_normal = comb[comb["normal"].values == True].copy()
        comb = comb[comb["normal"].values != True].copy()
    else:
        comb_normal = None

    weighted_epi = (
        comb[
            [
                "cohort",
                "patient_id",
                "tumor_epi_full",
                "tumor_epi_front",
                "tumor_epi_center",
            ]
        ]
        .groupby(["cohort", "patient_id"])
        .transform("sum")
    )
    weighted_epi.columns = [
        "tumor_epi_full_weight",
        "tumor_epi_front_weight",
        "tumor_epi_center_weight",
    ]
    comb["slide_count"] = comb.groupby(["cohort", "patient_id"])[
        "patient_id"
    ].transform("count")
    comb = pd.concat([comb, weighted_epi], axis=1)
    comb["tumor_epi_front_weight"] = (
        np.divide(comb["tumor_epi_front"].values, comb["tumor_epi_front_weight"].values)
        * comb["slide_count"].values
    )
    comb["tumor_epi_full_weight"] = (
        np.divide(comb["tumor_epi_full"].values, comb["tumor_epi_full_weight"].values)
        * comb["slide_count"].values
    )
    comb["tumor_epi_center_weight"] = (
        np.divide(comb["tumor_epi_center"], comb["tumor_epi_center_weight"].values)
        * comb["slide_count"].values
    )

    weighted = comb.copy()

    for r in ["r20", "r50", "r100", "r200", "r500"]:
        for ct in ["lym_", "eos_", "pla_", "neu_", "fil_", "mit_"]:
            for f in ["full_", "center_", "front_"]:
                try:
                    weighted[ct + f + r] = (
                        weighted[ct + f + r] * weighted["tumor_epi_" + f + "weight"]
                    )
                except:
                    pass
    for r in ["iel_ratio", "iel_pct_adj"]:
        weighted[r] = weighted[r] * weighted["tumor_epi_full_weight"]

    surv = (
        weighted.groupby(["cohort", "patient_id"], as_index=False)
        .mean(numeric_only=True)
        .copy()
    )
    surv.loc[
        surv.cohort == "berger",
        ["os", "os_status", "dfs", "dfs_status", "ttr", "ttr_status"],
    ] = surv.loc[
        surv.cohort == "berger",
        ["os_b", "os_status_b", "dfs_b", "dfs_status_b", "ttr_b", "ttr_status_b"],
    ].values

    surv["tnm_fac"] = surv.tnm.replace({1: "I", 2: "II", 3: "III", 4: "IV"})

    surv.loc[
        surv["os"] < 1, ["os", "os_status", "dfs", "dfs_status", "ttr", "ttr_status"]
    ] = np.NaN
    surv.loc[surv["os"] > 61, ["os", "os_status"]] = [61, 0]

    surv.loc[surv["dfs"] < 1, ["dfs", "dfs_status", "ttr", "ttr_status"]] = np.NaN
    surv.loc[surv["dfs"] > 61, ["dfs", "dfs_status", "ttr", "ttr_status"]] = [
        61,
        0,
        61,
        0,
    ]

    surv.loc[
        surv["os_b"] < 1,
        ["os_b", "os_status_b", "dfs_b", "dfs_status_b", "ttr_b", "ttr_status_b"],
    ] = np.NaN
    surv.loc[surv["os_b"] > 61, ["os_b", "os_status_b"]] = [61, 0]

    surv.loc[surv["dfs_b"] < 1, ["dfs_b", "dfs_status_b", "ttr_b", "ttr_status_b"]] = (
        np.NaN
    )
    surv.loc[surv["dfs_b"] > 61, ["dfs_b", "dfs_status_b", "ttr_b", "ttr_status_b"]] = [
        61,
        0,
        61,
        0,
    ]

    return comb, comb_normal, weighted, surv
