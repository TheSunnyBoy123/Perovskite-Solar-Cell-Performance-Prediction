"""Source-level decisions for the 17-paper recipe-flag pilot.

These are MODEL-ASSISTED judgements written by reading the cached Markdown full text
(sandbox/markdown). They are not human annotations. Every decision is one of:

  P(anchor, note)   supported positive: the paper's own devices/fabrication use the element.
                    The evidence is the sentence containing `anchor`, extracted verbatim
                    from the source text by build_audit.py (the anchor must occur in it).
  N(reason, note)   supported negative after review: the regex term (or an alias) is present
                    only in a non-qualifying context, or is absent from the full text.
  U(reason, ...)    unresolved: the source cannot settle the decision without adjudication.

Flags with no regex hit in the full text and no explicit entry are recorded as
N("NO_SOURCE_TERM") automatically. A flag that has regex hits but no explicit entry is an
error (build_audit.py refuses to run), so every visible hit has a recorded decision.

`borderline=True` marks a decision a human should check before it is used.
"""

def P(anchor, note, borderline=False):
    return {"label": "P", "anchor": anchor, "note": note, "borderline": borderline}


def N(reason, note="", borderline=False):
    return {"label": "N", "reason": reason, "note": note, "borderline": borderline}


def U(reason, anchor=None, note=""):
    return {"label": "U", "reason": reason, "anchor": anchor, "note": note, "borderline": True}


DECISIONS = {
    # ---------------------------------------------------------------- 10.1002/adfm.201604733
    "10.1002/adfm.201604733": {
        "pea_family": P("PEA is phenyl ethyl", "PEA-based quasi-2D (PEA)2(MA)n-1PbnBr3n+1 absorber used in the devices"),
        "hx_acid": P("with 100 mL of hydrobromic acid", "HBr is a reagent for MABr/PEABr precursor synthesis, not an additive; borderline for the flag definition", borderline=True),
        "htl_fk": P("cobalt(III) tris(bis(trifluoromethylsulfonyl)imide))", "FK209 Co(III) complex is the HTL dopant in the device recipe"),
        "htl_litfsi": P("sulfonimide lithium salt in acetonitrile", "LiTFSI in the HTL solution; regex misses the 'bis(trifluoromethylsulfonyl)imide lithium salt' wording"),
        "etl_ticl4": P("treated with TiCl4", "compact TiO2 layer treated with TiCl4 in device fabrication"),
        "lewis_base": N("LIGAND_NAME", "DMSO/DMF are solvents only; 4-tBP appears only inside the FK209 ligand name"),
        "htl_tbp": N("LIGAND_NAME", "4-tert-butylpyridine is a ligand of the FK209 cobalt complex, not a TBP additive"),
        "polymer": N("LITERATURE", "polymer HTMs discussed in the introduction"),
        "two_step": N("SPIN_PROGRAM", "'spin coated in two-step 1000 and 5000 rpm' is a two-speed programme"),
        "vacuum": N("ELECTRODE", "70 nm gold evaporated as the electrode; rotary evaporator and vacuum oven are process equipment"),
        "etl_dopant": N("COMPUTATIONAL", "'doped' cells appear only in DFT simulations"),
        "inorganic": N("GENERIC", "'organic-inorganic' perovskite terminology"),
        "cl_source": N("GENERIC", "chlorine only in the general ABX3 formula"),
    },
    # ---------------------------------------------------------------- 10.1002/aenm.201803587
    "10.1002/aenm.201803587": {
        "cl_source": P("addition of formamidinium chloride (FACl)", "FACl chloride additive in the precursor (title: chloride additives)"),
        "hx_acid": P("0.5 mL HCl (37%)", "HCl in the SnO2 chemical-bath deposition recipe (ETL); borderline for the flag definition", borderline=True),
        "cs": P("CsMAFA", "CsMAFA triple-cation reference composition and CsI solution added to the precursor"),
        "pea_family": U("DEFINITION_AMBIGUOUS", "1-adamantylamine hydrochloride",
                        "ADAHCl is a bulky amine hydrochloride passivator; the regex 'ammonium' matches only literature mentions. Is ADAHCl a PEA-family cation?"),
        "lewis_base": P("18 µL of 4-tert-butylpyridine", "4-tBP HTL additive (a pyridine Lewis base); the urea in the SnO2 bath is also present"),
        "htl_tbp": P("18 µL of 4-tert-butylpyridine", "4-tBP added to the spiro-OMeTAD solution"),
        "htl_litfsi": P("32 µL from the solution of bis(trifluoromethylsulfonyl)imide", "LiTFSI solution in the spiro-OMeTAD recipe; regex misses the wording"),
        "water_add": N("CLEANING", "DI water as the solvent of the CBD bath and substrate cleaning"),
        "polymer": N("ARTIFACT", "'ps' is picosecond pulse width; 'polymer' only in literature"),
        "carbon": N("LITERATURE", "fullerene listed in the literature introduction only"),
        "interface_pass": P("subsequently passivated by 1-adamantylamine hydrochloride", "ADAHCl passivation step after the FACl treatment"),
        "antisolvent": P("using an antisolvent technique", "title and method: antisolvent crystallisation with chloride additives"),
        "vacuum": N("LITERATURE", "thermal evaporation and sublimation of MACl/FACl discussed; no vacuum-deposited layer"),
        "htl_borate": N("ARTIFACT", "barium borate (BBO) NMR probe in the characterisation section"),
        "inorganic": N("GENERIC", "'organic-inorganic' terminology"),
        "n2_box": N("TEST_CONDITION", "nitrogen flow only during the MPP stability test"),
        "ambient": N("TEST_CONDITION", "air mass and humidity during characterisation and storage tests"),
    },
    # ---------------------------------------------------------------- 10.1007/s10118-017-1891-z
    "10.1007/s10118-017-1891-z": {
        "lewis_base": P("DMSO served as the additive", "DMSO used as an additive to tune the perovskite morphology"),
        "carbon": P("PC71BM film on perovskite layer", "PC71BM electron-transport layer; regex 'pcbm' misses the PC71BM alias"),
        "two_step": U("DEFINITION_AMBIGUOUS", "sequentially spin-coating a perovskite film",
                      "wet-on-wet sequential deposition of PC71BM onto the perovskite; not a two-step perovskite conversion"),
        "pea_family": N("ARTIFACT", "'ammonium' matches methylammonium iodide (MAI)"),
        "polymer": N("ARTIFACT", "'Polymer' is a department/journal name"),
        "antisolvent": N("ARTIFACT", "'quench' is photoluminescence quenching"),
        "vacuum": N("ELECTRODE", "silver electrode deposited in a vacuum chamber; DMSO evaporation is process physics"),
        "tandem": N("ARTIFACT", "'SI' is Supporting Information"),
        "mesoscopic": N("LITERATURE", "mesoporous TiO2 cited in the introduction only"),
        "n2_box": N("CLEANING", "substrates dried in a nitrogen flow; no glove box"),
        "scn": N("LITERATURE", "not present in the device; regex hit is not in this paper's recipe"),
        "undoped": N("GENERIC", "no undoped device layer"),
        "inorganic": N("GENERIC", "'organic-inorganic' terminology"),
    },
    # ---------------------------------------------------------------- 10.1016/j.electacta.2017.06.032
    "10.1016/j.electacta.2017.06.032": {
        "cl_source": P("Lead (II) Chloride", "PbCl2 precursor for the MAPbI3-xClx absorber"),
        "hx_acid": P("37% hydrochloric acid", "hydrothermal growth of the TiO2 nanorod ETL; regex misses 'hydrochloric'"),
        "etl_dopant": P("Based on Er-Yb Co-doped", "Er-Yb co-doped TiO2 nanorod electron-transport layer (title)"),
        "lewis_base": P("of 4-tert-butylpyridine and", "4-tBP HTL additive"),
        "htl_tbp": P("of 4-tert-butylpyridine and", "4-tBP added to the spiro-OMeTAD solution"),
        "htl_litfsi": P("Li-TFSI solution (520 mg", "Li-TFSI added to the spiro-OMeTAD solution"),
        "undoped": P("Both the solar cells based on un-doped", "undoped-TiO2 control cells fabricated and stored alongside the Er-Yb cells"),
        "n2_box": P("in an argon-filled glove box", "perovskite precursor spin-coated in an argon glove box"),
        "water_add": N("CLEANING", "DI water for cleaning and as the hydrothermal solvent"),
        "polymer": N("ARTIFACT", "'Polymer' is the supplier name"),
        "alkali": N("LITERATURE", "rubidium incorporation cited in the references"),
        "pea_family": N("ARTIFACT", "'ammonium' matches methylammonium iodide"),
        "vacuum": N("ELECTRODE", "80 nm gold thermally evaporated as the electrode"),
        "ambient": N("TEST_CONDITION", "storage in ambient air without encapsulation"),
        "interface_pass": N("REFERENCE", "'modification' appears in the reference list only"),
        "tandem": N("REFERENCE", "tandem in the reference list only"),
        "mesoscopic": N("REFERENCE", "mesoporous TiO2 in the reference list; the device uses TiO2 nanorods"),
    },
    # ---------------------------------------------------------------- 10.1016/j.joule.2018.06.009
    "10.1016/j.joule.2018.06.009": {
        "alkali": P("with the composition Rb0.05", "Rb0.05 in the device composition of the lab-fabricated n-i-p cells"),
        "cs": P("with the composition Rb0.05", "Cs0.05 in the device composition; regex 'cs' does not match 'Cs0.05'"),
        "carbon": P("ITO/SnO2/C60/perovskite", "C60 electron-transport layer in the lab-fabricated n-i-p cell"),
        "antisolvent": U("DEFINITION_AMBIGUOUS", "complex-assisted gas quenching (CAGQ) approach",
                         "gas quench is a quench method, not an antisolvent; regex 'quench' matches; whether the flag definition includes it is for adjudication"),
        "n2_box": U("DEFINITION_AMBIGUOUS", "the application of a nitrogen flow expedites",
                    "N2 gas stream during film drying; not a glove box; flag definition for a gas-quench step needs adjudication"),
        "pea_family": N("REFERENCE", "'Bai' is an author name"),
        "htl_cu": N("LITERATURE", "copper indium gallium selenide (CIGS) only"),
        "htl_borate": N("ARTIFACT", "'corroborated' contains 'borate'"),
        "interface_pass": N("GENERIC", "'modifications' and 'posttreatments' discussed in the review text"),
        "vacuum": N("LITERATURE", "vacuum deposition discussed as a review topic"),
        "inorganic": N("LITERATURE", "organic-inorganic halide perovskite terminology"),
        "tandem": N("LITERATURE", "tandem applications discussed in the text"),
        "ambient": N("GENERIC", "humidity and 'air knife' in review discussion"),
        "cl_source": N("GENERIC", "chlorine only in the general ABX3 formula"),
    },
    # ---------------------------------------------------------------- 10.1016/j.jpowsour.2018.08.070
    "10.1016/j.jpowsour.2018.08.070": {
        "undoped": P("synthesized for dopant-free PSCs", "BDT2FMeDPA dopant-free hole-transport material is the device HTM"),
        "mesoscopic": P("mesoporous-TiO2/ perovskite", "compact TiO2 / mesoporous TiO2 scaffold in the device stack"),
        "etl_dopant": N("ARTIFACT", "'FTO' is fluorine-doped tin oxide substrate; 'doped' ones refers to doped HTMs"),
        "antisolvent": N("ARTIFACT", "'quench' is photoluminescence quenching"),
        "vacuum": N("PROCESS_DRYING", "'vacuum-free carbon cathode' describes the absence of vacuum processing"),
        "two_step": N("CHEMISTRY", "'two-step' refers to the HTM synthesis"),
        "interface_pass": N("GENERIC", "'modification' refers to molecular structure"),
        "polymer": N("LITERATURE", "polymer HTMs in the introduction"),
        "inorganic": N("LITERATURE", "inorganic HTMs in the introduction"),
        "ambient": N("TEST_CONDITION", "air stability tests"),
        "cl_source": N("CHEMISTRY", "CH2Cl2 solvent for absorption spectra"),
    },
    # ---------------------------------------------------------------- 10.1016/j.nanoen.2018.10.011
    "10.1016/j.nanoen.2018.10.011": {
        "cs": P("CsI (0.066 M)", "CsI in the mixed-cation precursor"),
        "carbon": P("perovskite/PCBM/Ag architecture", "PCBM electron-transport layer in the p-i-n device; PCBM and ITIC also used as anti-solvents"),
        "antisolvent": P("as anti-solvent and dripped onto", "PCBM and ITIC dissolved in chlorobenzene and dripped onto the film"),
        "interface_pass": P("to passivate the defects", "PCBM/ITIC anti-solvent treatment described as defect passivation"),
        "n2_box": P("N2-protected glove box", "perovskite films prepared in an N2 glove box"),
        "ambient": P("10 min in ambient air", "PEDOT:PSS layer annealed in ambient air during device fabrication"),
        "lewis_base": N("SOLVENT", "DMF and DMSO solvents only"),
        "hx_acid": N("CHEMISTRY", "HI is a thermal decomposition product, not a recipe ingredient"),
        "pea_family": N("ARTIFACT", "'ammonium' matches methylammonium"),
        "polymer": N("ARTIFACT", "'Polymer Light Technology' is a supplier name"),
        "water_add": N("SOLVENT", "ultrapure H2O used to dilute the SnO2 colloid; a diluent, not an additive"),
        "htl_cu": N("LITERATURE", "copper indium gallium selenide only"),
        "undoped": N("ARTIFACT", "'pristine' refers to as-prepared (unannealed) films"),
        "vacuum": N("GENERIC", "vacuum permittivity and vacuum level in the physics text"),
        "two_step": N("GENERIC", "'sequentially' describes decomposition order"),
        "tandem": N("ARTIFACT", "'Silicon' is the affiliation name"),
        "inorganic": N("LITERATURE", "organic-inorganic terminology"),
        "flexible": N("REFERENCE", "'flexible' in author biographies"),
    },
    # ---------------------------------------------------------------- 10.1021/acs.jpcc.5b06269
    "10.1021/acs.jpcc.5b06269": {
        "carbon": P("bathocuproine (BCP)/Al layers have been added", "PCBM/BCP/Al top contact of the compared devices"),
        "pea_family": N("ARTIFACT", "methylammonium iodide (MAI) only"),
        "lewis_base": N("SOLVENT", "DMSO as part of the GBL/DMSO solvent mixture"),
        "polymer": N("LITERATURE", "polymer hole-transport materials in references"),
        "interface_pass": N("LITERATURE", "PCBM passivation cited from literature"),
        "vacuum": N("GENERIC", "water-vapour effects on the film"),
        "n2_box": N("TEST_CONDITION", "nitrogen glove box during light-soaking only"),
        "ambient": N("TEST_CONDITION", "photodegradation studies in ambient air"),
        "htl_cu": N("REFERENCE", "copper-doped NiO in the reference list"),
        "etl_dopant": N("REFERENCE", "copper-doped NiO in the reference list"),
        "undoped": N("ARTIFACT", "'pristine' refers to film absorbance; dopant-free only in references"),
        "tandem": N("ARTIFACT", "'Si' is a photodiode reference"),
        "inorganic": N("LITERATURE", "organic-inorganic terminology"),
    },
    # ---------------------------------------------------------------- 10.1021/acs.jpcc.8b01026
    "10.1021/acs.jpcc.8b01026": {
        "two_step": P("globally modify the two-step deposition process", "BMII additive introduced into the two-step (sequential) MAPbI3 deposition"),
        "interface_pass": N("GENERIC", "global modification by BMII additive, not an interface treatment; BMII is not in the flag ontology"),
        "cl_source": N("LITERATURE", "CH3NH3Cl in the reference list"),
        "scn": N("REFERENCE", "pseudohalide in the reference list"),
        "hx_acid": N("REFERENCE", "HCl in the reference list"),
        "cs": N("REFERENCE", "CsPbI3 in the reference list"),
        "pea_family": N("LITERATURE", "tetra-n-butylammonium and MAI in discussion"),
        "lewis_base": N("LITERATURE", "DMSO and pyridine discussed as cited additives"),
        "carbon": N("REFERENCE", "fullerene in the reference list"),
        "undoped": N("ARTIFACT", "'pristine' refers to pristine PbI2 films"),
        "vacuum": N("LITERATURE", "vapour-assisted processes discussed in the literature"),
        "solvent_anneal": N("REFERENCE", "solvent annealing in the reference list"),
        "tandem": N("REFERENCE", "silicon tandem in the reference list"),
        "inorganic": N("GENERIC", "'inorganic' in the framework description"),
        "mesoscopic": N("REFERENCE", "mesoporous PbI2 in the reference list"),
        "ambient": N("REFERENCE", "ambient conditions in the reference list"),
    },
    # ---------------------------------------------------------------- 10.1021/acs.jpclett.5b01716
    "10.1021/acs.jpclett.5b01716": {
        "vacuum": P("Vacuum deposition techniques are used to prepare", "mixed iodide-bromide MAPb perovskite diodes prepared by vacuum deposition"),
        "carbon": P("as the electron transport material, capped", "PCBM electron-transport layer in the diode stack"),
        "polymer": P("a hole transporting polymer and covering", "PEDOT:PSS hole-transport polymer layer; borderline for a polymer-as-additive definition", borderline=True),
        "two_step": U("DEFINITION_AMBIGUOUS", "via an intermediate double layer of the pure",
                      "sequential vacuum deposition of iodide then bromide layers; not a solution two-step"),
        "pea_family": N("ARTIFACT", "methylammonium lead perovskite; 'ammonium' matches MA"),
        "cl_source": N("LITERATURE", "chloride in the introduction"),
        "etl_dopant": N("ARTIFACT", "PEDOT:PSS 'doped with' PSS; not an ETL dopant"),
        "interface_pass": N("GENERIC", "'modifies' and 'modifications' refer to bandgap and chemistry"),
        "antisolvent": N("ARTIFACT", "'quenching' refers to exciton quenching"),
        "tandem": N("LITERATURE", "tandem applications in the introduction"),
        "inorganic": N("LITERATURE", "organic-inorganic terminology"),
    },
    # ---------------------------------------------------------------- 10.1038/ncomms13503
    "10.1038/ncomms13503": {
        "hx_acid": P("reaction of PbI2 with excess HI to generate HPbI3", "HI is a reagent in the NABR perovskite synthesis route used for the devices"),
        "two_step": P("alternative two-step nonstoichiometric acid-base reaction route", "two-step NABR synthesis route for the device perovskite"),
        "lewis_base": P("35 µl 4-tert-butylpyridine", "4-tBP additive in the spiro-OMeTAD solution"),
        "htl_tbp": P("35 µl 4-tert-butylpyridine", "4-tBP added to the spiro-OMeTAD solution"),
        "htl_litfsi": P("25 µl LiTFSI solution", "LiTFSI added to the spiro-OMeTAD solution"),
        "mesoscopic": P("was immersed in aqueous 30 mM TiCl", "mesoporous TiO2 scaffold (TiO2 paste) in the device stack"),
        "etl_ticl4": P("was immersed in aqueous 30 mM TiCl", "TiCl4 treatment of the mesoporous TiO2; regex misses 'TiCl $_4$' with markup"),
        "n2_box": P("All device fabrication procedures were carried out", "device fabrication in an N2-purged glove box"),
        "antisolvent": U("DEFINITION_AMBIGUOUS", "adding antisolvents",
                         "diethyl ether named as an antisolvent in a comparison; also used in the synthesis route"),
        "interface_pass": N("PROCESS_DRYING", "ethanol washing to remove excess HI; a process wash, not a passivation layer", borderline=True),
        "pea_family": N("ARTIFACT", "'methylammonium iodide' is MA"),
        "ambient": N("TEST_CONDITION", "humidity stability tests"),
        "vacuum": N("GENERIC", "CH3NH2 vapour exposure and vapour terms"),
        "polymer": N("LITERATURE", "carbon nanotube/polymer composite in the reference list"),
        "carbon": N("REFERENCE", "carbon nanotube in the reference list"),
        "etl_dopant": N("ARTIFACT", "'F-doped SnO2' is the FTO substrate"),
        "water_add": N("SOLVENT", "DI water rinse after TiCl4 treatment"),
        "scn": N("REFERENCE", "pseudohalide in the reference list"),
        "cl_source": N("LITERATURE", "chloride in the introduction"),
        "tandem": N("LITERATURE", "silicon cells in the introduction"),
        "inorganic": N("GENERIC", "organic-inorganic terminology"),
    },
    # ---------------------------------------------------------------- 10.1038/ncomms15330
    "10.1038/ncomms15330": {
        "acetate": P("15 mol% magnesium acetate tetrahydrate", "Mg-acetate-doped nickel acetylacetonate precursor for the sprayed NiO hole layer in the device process"),
        "etl_dopant": P("containing N-doped graphene", "N-doped graphene in the electron-extraction layer"),
        "carbon": P("the fullerene derivative phenyl-C61-butyric acid methyl ester (PCBM)", "graphene/PCBM electron-extraction layer"),
        "interface_pass": P("as the interlayer between PCBM and silver", "LiF and Ca interlayers compared with the CQD interlayer in the fabricated devices; borderline", borderline=True),
        "n2_box": P("were done in glove box", "all device processes performed in a glove box"),
        "hx_acid": N("ETCH", "2 M HCl used to etch the FTO pattern, not a device layer"),
        "pea_family": N("ARTIFACT", "MA/FA mentioned generically; 'ammonium' in references"),
        "cl_source": N("LITERATURE", "chloride in the reference list"),
        "lewis_base": N("REFERENCE", "phosphonic acid in the reference list"),
        "polymer": N("REFERENCE", "polymer solar cells in the reference list; 'ps' is picosecond pulse width"),
        "undoped": N("ARTIFACT", "'pristine' refers to PCBM films; dopant-free only in references"),
        "antisolvent": N("ARTIFACT", "'quench' is PL quenching"),
        "vacuum": N("ELECTRODE", "Ag or Au/Ag electrode evaporated under high vacuum"),
        "tandem": N("LITERATURE", "tandem in the introduction; 'silicon' is the reference cell"),
        "inorganic": N("LITERATURE", "organic-inorganic terminology"),
        "ambient": N("TEST_CONDITION", "air-mass illumination and air exposure tests"),
    },
    # ---------------------------------------------------------------- 10.1039/c7se00100b
    "10.1039/c7se00100b": {
        "cs": P("a well-defined CsI crystalline film", "CsI film forms the first step of the Cs2SnI6 two-step synthesis"),
        "lead_free": P("new class of tin-based and air-stable", "lead-free tin-based Cs2SnI6 absorber"),
        "two_step": P("A two-step solution synthesis method has been developed", "two-step solution synthesis of the Cs2SnI6 films"),
        "etl_ticl4": P("TiCl4 aqueous solution has been applied", "TiCl4 treatment of the E-sprayed TiO2 electron-transport layer"),
        "etl_dopant": P("Sn doped TiO<sub>2</sub> layer as a photo-anode", "Sn-doped TiO2 photoanode layer"),
        "inorganic": P("for Cs<sub>2</sub>SnI<sub>6</sub> thin films", "all-inorganic Cs2SnI6 absorber; borderline for the flag definition", borderline=True),
        "polymer": U("DEFINITION_AMBIGUOUS", "functions as a polymer host medium for ion transport",
                     "succinonitrile is called a polymer host but is a plastic crystal; electrolyte composition not confirmed from the text"),
        "water_add": N("SOLVENT", "DI water as the solvent for the CsI spray solution and for the TiCl4 dilution"),
        "scn": N("LITERATURE", "CuSCN discussed as another hole material"),
        "htl_cu": N("LITERATURE", "CuSCN discussed in the introduction"),
        "htl_litfsi": N("REFERENCE", "LiTFSI in the reference list"),
        "cl_source": N("REFERENCE", "Cs2SnX6 (X = Cl, Br, I) in the reference list"),
        "undoped": N("FILM_ONLY", "undoped E-sprayed TiO2 characterised as a film"),
        "interface_pass": N("GENERIC", "Sn 'modification' is doping, not an interface treatment"),
        "vacuum": N("TEST_CONDITION", "vacuum heat treatment used in a stability test"),
        "flexible": N("ARTIFACT", "'flexible' describes the free-standing HTM film"),
        "tandem": N("ARTIFACT", "silicon reference cell and 'SI' supplementary info"),
        "ambient": N("TEST_CONDITION", "'air-stable' material claim and ambient stability tests"),
        "n2_box": N("AFFILIATION", "'Argonne' affiliation"),
    },
    # ---------------------------------------------------------------- 10.1039/c8ta07968d
    "10.1039/c8ta07968d": {
        "cs": P("toward CsPb(", "all-inorganic CsPb(IxBr1-x)3 absorber"),
        "lewis_base": P("as an additive in the precursor solution", "DMSO used as an additive in the CsBr/PbX2 precursor solution"),
        "htl_tbp": P("additives of Li-TFSI, FK209 and TBP", "TBP additive in the spiro-OMeTAD solution"),
        "htl_litfsi": P("additives of Li-TFSI, FK209 and TBP", "Li-TFSI additive in the spiro-OMeTAD solution"),
        "htl_fk": P("additives of Li-TFSI, FK209 and TBP", "FK209 additive in the spiro-OMeTAD solution"),
        "interface_pass": U("DEFINITION_AMBIGUOUS", "spin-coated with LiTFSI acetonitrile solution",
                            "LiTFSI spin-coated on the TiCl4-treated TiO2 before the perovskite; surface treatment or only a dopant?"),
        "two_step": P("a two-step multiple spin coating method has been developed", "two-step multiple spin-coating deposition of the all-inorganic film"),
        "mesoscopic": P("give a 100 nm-thick", "100 nm mesoporous TiO2 scaffold in the device"),
        "inorganic": P("films for high-performance inorganic perovskite solar cells", "all-inorganic CsPb(IxBr1-x)3 perovskite"),
        "n2_box": P("two-step deposition method in a glovebox", "perovskite deposited in a glove box"),
        "vacuum": N("PROCESS_DRYING", "vacuum step removes excess DMF; the Au electrode is thermally evaporated", borderline=True),
        "alkali": N("LITERATURE", "Rb incorporation cited in the introduction"),
        "pea_family": N("REFERENCE", "'Bai' is an author name"),
        "water_add": N("CLEANING", "DI water in substrate cleaning"),
        "solvent_anneal": N("LITERATURE", "solvent vapour annealing mentioned as a mechanism comparison"),
        "tandem": N("ARTIFACT", "silicon reference solar cell"),
        "ambient": N("TEST_CONDITION", "'98% PCE under ambient conditions' stability claim"),
    },
    # ---------------------------------------------------------------- 10.1039/c9ta01893j
    "10.1039/c9ta01893j": {
        "pea_family": P("Guanidinium induced phase separated perovskite layer", "guanidinium (GA) bulky A-site cation is the paper's dopant; regex 'guanidin' is the intended match"),
        "cs": P("$Cs_{0.05}(FA_{0.83}", "Cs0.05 in the quadruple-cation absorber; regex does not match 'Cs0.05'"),
        "carbon": P("FTO/SnO2/perovskite/PCBM/Ag", "PCBM in the electron-only devices fabricated by the authors; regex misses 'PC71BM' wording only in other cases"),
        "lewis_base": P("28.8 $\\mu$ L of 4-tert-butylpyridine", "4-tBP HTL additive; the DMF/DMSO solvents are not counted"),
        "htl_tbp": P("28.8 $\\mu$ L of 4-tert-butylpyridine", "4-tBP added to the spiro-OMeTAD solution"),
        "htl_litfsi": P("17.5 $\\mu$ L of lithium-bis(trifluoromethanesulfonyl)imide", "LiTFSI added to the spiro-OMeTAD solution"),
        "n2_box": P("transferred to a nitrogen filled glove box", "perovskite precursor prepared in a nitrogen glove box"),
        "ava": N("LITERATURE", "aminovaleric acid cited in the introduction only"),
        "polymer": N("ARTIFACT", "'Polymer' is the supplier name"),
        "etl_dopant": N("GENERIC", "'GA doped' refers to the absorber, not an ETL dopant"),
        "interface_pass": N("GENERIC", "'passivation' by the δ-FAPbI3 phase is a bulk phase effect, not an interface treatment"),
        "vacuum": N("ELECTRODE", "Au electrode thermally evaporated under high vacuum"),
        "tandem": N("ARTIFACT", "reference Si solar cell"),
        "inorganic": N("LITERATURE", "all-inorganic perovskites in the introduction; the absorber is hybrid"),
        "ambient": N("TEST_CONDITION", "stability tests at 25% and 40% humidity"),
    },
    # ---------------------------------------------------------------- 10.1063/1.4896779
    "10.1063/1.4896779": {
        "cl_source": P("using chlorine precursor", "CH3NH3PbI3-xClx absorber made with a chlorine precursor (planar p-i-n-type cells)"),
        "lewis_base": P("28.8 $\\mu$ l of 4-tert-butylpyridine", "4-tBP HTL additive"),
        "htl_tbp": P("28.8 $\\mu$ l of 4-tert-butylpyridine", "4-tBP added to the spiro-OMeTAD solution"),
        "htl_litfsi": P("lithium bis-(trifluoromethylsulfonyl)imide in acetonitrile", "LiTFSI added to the HTM solution"),
        "n2_box": P("device fabrication was carried out under controlled atmospheric conditions", "controlled atmosphere (humidity < 0.5 ppm) for device fabrication"),
        "ambient": P("under an air stream", "film annealed at 100 °C under an air stream during processing", borderline=True),
        "etl_dopant": U("DEFINITION_AMBIGUOUS", "doped TiO<sub>2</sub> compact layers",
                        "n-doped TiO2 compact layer is discussed in a band/physics context; the device ETL doping is not confirmed"),
        "mesoscopic": N("PLANAR_DEVICE", "nanoporous scaffold used only as a comparison; the device is planar"),
        "hx_acid": N("ETCH", "2 M HCl used to etch the FTO pattern"),
        "water_add": N("CLEANING", "DI water and ethanol in substrate cleaning"),
        "cs": N("REFERENCE", "CsBr addition in the reference list"),
        "pea_family": N("NO_SOURCE_TERM", "no regex hit in the full text; the JSON flag has no source support"),
        "undoped": N("GENERIC", "'undoped absorber' appears in the physics discussion"),
        "interface_pass": N("GENERIC", "'modify' describes ionic bonding in the physics discussion"),
        "vacuum": N("ELECTRODE", "60 nm gold thermally evaporated as the electrode"),
        "two_step": N("REFERENCE", "two-step interdiffusion in the reference list"),
        "tandem": N("ARTIFACT", "reference Si solar cell"),
    },
    # ---------------------------------------------------------------- 10.1088/1361-6463/ab1626
    "10.1088/1361-6463/ab1626": {
        "cl_source": P("PbI2 and PbCl2 in 1:1:0.1 molar ratio", "PbCl2 added to the CH3NH3I/PbI2 precursor (MAPbIxCl3-x)"),
        "hx_acid": P("hydroiodic acid (HI, 57% conc. in H2O)", "HI used to synthesise the MAI precursor; reagent, not an additive; borderline", borderline=True),
        "carbon": P("PC<sub>71</sub>BM/ZnO/Al", "PC71BM electron-transport layer in the p-i-n device; regex 'pcbm' misses PC71BM"),
        "polymer": P("PEDOT: PSS (Heraeus Clevios", "PEDOT:PSS hole-transport polymer layer; borderline for a polymer-as-additive definition", borderline=True),
        "pea_family": N("ARTIFACT", "'ammonium' matches methylammonium iodide; 'Multifunctional Ammonium Salt' is a conference title in a link list"),
        "lewis_base": N("REFERENCE", "aminopyridine in the reference list"),
        "water_add": N("SOLVENT", "H2O as HI solvent and DI water cleaning"),
        "etl_dopant": N("REFERENCE", "La-doped BaSn3 in the reference list"),
        "interface_pass": N("REFERENCE", "passivation reviews in the reference list"),
        "antisolvent": N("ARTIFACT", "'quench' refers to trap-assisted recombination"),
        "vacuum": N("LITERATURE", "vacuum deposition in the introduction; rotary evaporation and vacuum oven drying"),
        "two_step": N("REFERENCE", "two-step fabrication in the reference list"),
        "tandem": N("ARTIFACT", "silicon reference solar cell"),
        "lead_free": N("REFERENCE", "tin and germanium perovskites in the reference list"),
        "inorganic": N("LITERATURE", "organic-inorganic hybrid perovskite terminology"),
        "ambient": N("REFERENCE", "air degradation of organic solar cells in the reference list"),
        "cs": N("REFERENCE", "CsPbBr3 in the reference list"),
    },
}


def decision_for(doi, flag):
    """Explicit decision for (doi, flag) or None."""
    return DECISIONS.get(doi, {}).get(flag)
