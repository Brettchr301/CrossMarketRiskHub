from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class UniverseTicker:
    ticker: str
    commodity_type: str
    country: str
    sector: str


def _build(items: Iterable[tuple[str, str, str, str]]) -> list[UniverseTicker]:
    out: list[UniverseTicker] = []
    seen: set[str] = set()
    for ticker, commodity_type, country, sector in items:
        t = ticker.upper().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(
            UniverseTicker(
                ticker=t,
                commodity_type=commodity_type,
                country=country,
                sector=sector,
            )
        )
    return out


GLOBAL_COMMODITY_UNIVERSE: list[UniverseTicker] = _build(
    [
        # ═══════════════════════════════════════════════════════════════════
        # FOCUSED UNIVERSE: Only sectors with POSITIVE vs-sector alpha
        # Pruned: shipping (-42 to -58%), coal (-54%), base_metals (-20%),
        #         uranium (-16%), agri (-8%), oil_services (0%)
        # Kept + expanded: oil_upstream (+9.5%), oil_refining (+9%),
        #         midstream (+13%), lithium (+51%), rare_earths (+57%),
        #         precious_metals (kept smaller — high Sharpe 2.8)
        # ═══════════════════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════════════
        # OIL & GAS UPSTREAM — +9.48% α vs XOP (416 trades, most reliable)
        # MASSIVELY expanded micro/small caps (highest alpha potential)
        # ══════════════════════════════════════════════════════════════

        # ── US micro-cap E&P (<$500M) — NEW, doubled ──
        ("RING", "oil_gas_upstream", "US", "E&P"),  # Ring Energy ~$400M
        ("REI", "oil_gas_upstream", "US", "E&P"),  # Ring Energy (alt)
        ("BATL", "oil_gas_upstream", "US", "E&P"),  # Battalion Oil
        ("TALO", "oil_gas_upstream", "US", "E&P"),  # Talos Energy
        ("NEXT", "oil_gas_upstream", "US", "E&P"),  # NextDecade Corp
        ("GPOR", "oil_gas_upstream", "US", "E&P"),  # Gulfport Energy
        ("CPG", "oil_gas_upstream", "CA", "E&P"),  # Crescent Point (US-listed)
        ("SD", "oil_gas_upstream", "US", "E&P"),  # SandRidge Energy
        ("CTRA", "oil_gas_upstream", "US", "E&P"),  # Coterra (was Cabot)
        ("EGY", "oil_gas_upstream", "US", "E&P"),  # Vaalco Energy (micro, Africa ops)
        ("ESTE", "oil_gas_upstream", "US", "E&P"),  # Earthstone
        ("VTS", "oil_gas_upstream", "US", "E&P"),  # Vitesse Energy
        ("PHX", "oil_gas_upstream", "US", "Royalty"),  # PHX Minerals
        ("SBOW", "oil_gas_upstream", "US", "E&P"),  # SilverBow Resources
        ("KOS", "oil_gas_upstream", "US", "E&P"),  # Kosmos Energy (offshore Africa)
        ("TXO", "oil_gas_upstream", "US", "E&P"),  # TXO Partners
        ("EPM", "oil_gas_upstream", "US", "E&P"),  # Evolution Petroleum
        ("CTRA", "oil_gas_upstream", "US", "E&P"),  # Coterra Energy

        # ── US small-cap E&P ($500M-$2B) — existing + expanded ──
        ("AR", "oil_gas_upstream", "US", "E&P"),
        ("RRC", "oil_gas_upstream", "US", "E&P"),
        ("SM", "oil_gas_upstream", "US", "E&P"),
        ("MTDR", "oil_gas_upstream", "US", "E&P"),
        ("MUR", "oil_gas_upstream", "US", "E&P"),
        ("NOG", "oil_gas_upstream", "US", "E&P"),
        ("PR", "oil_gas_upstream", "US", "E&P"),
        ("CHRD", "oil_gas_upstream", "US", "E&P"),
        ("CRGY", "oil_gas_upstream", "US", "E&P"),
        ("CRC", "oil_gas_upstream", "US", "E&P"),
        ("MGY", "oil_gas_upstream", "US", "E&P"),
        ("VTLE", "oil_gas_upstream", "US", "E&P"),
        ("CDEV", "oil_gas_upstream", "US", "E&P"),
        ("TELL", "oil_gas_upstream", "US", "LNG"),
        # NEW small-cap US E&P
        ("GRNT", "oil_gas_upstream", "US", "E&P"),  # Granite Point Mortgage (energy pivot)
        ("CLNE", "oil_gas_upstream", "US", "E&P"),  # Clean Energy Fuels
        ("REPX", "oil_gas_upstream", "US", "Royalty"),  # Riley Exploration Permian
        ("ESTE", "oil_gas_upstream", "US", "E&P"),
        ("VIST", "oil_gas_upstream", "AR", "E&P"),  # Vista Energy (Vaca Muerta)
        ("VET", "oil_gas_upstream", "CA", "E&P"),

        # ── US mid/large E&P (existing) ──
        ("OXY", "oil_gas_upstream", "US", "Integrated"),
        ("APA", "oil_gas_upstream", "US", "E&P"),
        ("DVN", "oil_gas_upstream", "US", "E&P"),
        ("EOG", "oil_gas_upstream", "US", "E&P"),
        ("FANG", "oil_gas_upstream", "US", "E&P"),

        # ── Canada E&P — existing + new micro/small ──
        ("SU", "oil_gas_upstream", "CA", "Integrated"),
        ("CNQ", "oil_gas_upstream", "CA", "E&P"),
        ("CVE", "oil_gas_upstream", "CA", "E&P"),
        ("BTE.TO", "oil_gas_upstream", "CA", "E&P"),
        ("PEY.TO", "oil_gas_upstream", "CA", "E&P"),
        ("TVE.TO", "oil_gas_upstream", "CA", "E&P"),
        ("ATH.TO", "oil_gas_upstream", "CA", "E&P"),
        ("TOU.TO", "oil_gas_upstream", "CA", "E&P"),
        ("ARX.TO", "oil_gas_upstream", "CA", "E&P"),
        ("BIR.TO", "oil_gas_upstream", "CA", "E&P"),
        ("NVA.TO", "oil_gas_upstream", "CA", "E&P"),
        ("PSK.TO", "oil_gas_upstream", "CA", "E&P"),
        ("KEL.TO", "oil_gas_upstream", "CA", "E&P"),
        ("WCP.TO", "oil_gas_upstream", "CA", "E&P"),
        ("BTE", "oil_gas_upstream", "CA", "E&P"),
        # NEW Canada micro/small E&P
        ("SDE.TO", "oil_gas_upstream", "CA", "E&P"),  # Spartan Delta ~$300M
        ("CJ.TO", "oil_gas_upstream", "CA", "E&P"),  # Cardinal Energy
        ("FRU.TO", "oil_gas_upstream", "CA", "Royalty"),  # Freehold Royalties
        ("TPZ.TO", "oil_gas_upstream", "CA", "Royalty"),  # Topaz Energy
        ("POU.TO", "oil_gas_upstream", "CA", "E&P"),  # Paramount Resources
        ("JOY.TO", "oil_gas_upstream", "CA", "E&P"),  # Journey Energy micro
        ("SGY.TO", "oil_gas_upstream", "CA", "E&P"),  # Surge Energy micro
        ("GXE.TO", "oil_gas_upstream", "CA", "E&P"),  # Gear Energy micro
        ("HWX.TO", "oil_gas_upstream", "CA", "E&P"),  # Headwater Exploration
        ("AAV.TO", "oil_gas_upstream", "CA", "E&P"),  # Advantage Energy
        ("RBY.TO", "oil_gas_upstream", "CA", "E&P"),  # Rubellite Energy micro

        # ── International Oil E&P (existing + new micro/small) ──
        ("WDS", "oil_gas_upstream", "AU", "Integrated"),
        ("EQNR", "oil_gas_upstream", "NO", "Integrated"),
        ("TTE", "oil_gas_upstream", "FR", "Integrated"),
        ("SHEL", "oil_gas_upstream", "UK", "Integrated"),
        ("PBR", "oil_gas_upstream", "BR", "Integrated"),
        ("YPF", "oil_gas_upstream", "AR", "Integrated"),
        ("EC", "oil_gas_upstream", "CO", "Integrated"),
        ("EQNR.OL", "oil_gas_upstream", "NO", "Integrated"),
        ("AKRBP.OL", "oil_gas_upstream", "NO", "E&P"),
        ("VAR.OL", "oil_gas_upstream", "NO", "E&P"),
        ("OMV.VI", "oil_gas_upstream", "AT", "Integrated"),
        ("GALP.LS", "oil_gas_upstream", "PT", "Integrated"),
        ("ENI.MI", "oil_gas_upstream", "IT", "Integrated"),
        ("ENOG.L", "oil_gas_upstream", "UK", "E&P"),
        ("HBR.L", "oil_gas_upstream", "UK", "E&P"),
        ("TLW.L", "oil_gas_upstream", "UK", "E&P"),
        ("PETR4.SA", "oil_gas_upstream", "BR", "Integrated"),
        ("PRIO3.SA", "oil_gas_upstream", "BR", "E&P"),
        ("RECV3.SA", "oil_gas_upstream", "BR", "E&P"),
        # NEW international micro/small E&P
        ("ITM.L", "oil_gas_upstream", "UK", "E&P"),  # Ithaca Energy (North Sea micro)
        ("ENQ.L", "oil_gas_upstream", "UK", "E&P"),  # EnQuest (North Sea micro)
        ("PMO.L", "oil_gas_upstream", "UK", "E&P"),  # Premier Miton micro
        ("PTAL.L", "oil_gas_upstream", "UK", "E&P"),  # Petro Tal (Peru)
        ("SQZ.L", "oil_gas_upstream", "UK", "E&P"),  # Serica Energy micro
        ("CAPD.L", "oil_gas_upstream", "UK", "E&P"),  # Capital Ltd
        ("STX.AX", "oil_gas_upstream", "AU", "E&P"),  # Strike Energy micro
        ("BPT.AX", "oil_gas_upstream", "AU", "E&P"),  # Beach Energy small
        ("KAR.AX", "oil_gas_upstream", "AU", "E&P"),  # Karoon Energy small
        ("CVN.AX", "oil_gas_upstream", "AU", "E&P"),  # Carnarvon Energy micro
        ("COE.AX", "oil_gas_upstream", "AU", "E&P"),  # Cooper Energy micro
        ("STO.AX", "oil_gas_upstream", "AU", "Integrated"),  # Santos

        # ══════════════════════════════════════════════════════════════
        # OIL REFINING — +9.03% α vs CRAK (103 trades)
        # Expanded with micro/small refiners
        # ══════════════════════════════════════════════════════════════
        ("DINO", "oil_refining", "US", "Refining"),
        ("PBF", "oil_refining", "US", "Refining"),
        ("MPC", "oil_refining", "US", "Refining"),
        ("VLO", "oil_refining", "US", "Refining"),
        ("PSX", "oil_refining", "US", "Refining"),
        ("DK", "oil_refining", "US", "Refining"),
        ("REP.MC", "oil_refining", "ES", "Integrated"),
        ("PKN.WA", "oil_refining", "PL", "Refining"),
        ("MOL.BU", "oil_refining", "HU", "Integrated"),
        ("028300.KS", "oil_refining", "KR", "Refining"),
        ("096770.KS", "oil_refining", "KR", "Refining"),
        ("267250.KS", "oil_refining", "KR", "Refining"),
        # NEW micro/small refiners
        ("PARR", "oil_refining", "US", "Refining"),  # Par Pacific small
        ("CVI", "oil_refining", "US", "Refining"),  # CVR Energy small
        ("CLMT", "oil_refining", "US", "Refining"),  # Calumet Specialty small
        ("CAPL", "oil_refining", "US", "Refining"),  # CrossAmerica Partners
        ("DINO", "oil_refining", "US", "Refining"),  # HF Sinclair
        ("NS", "oil_refining", "US", "Refining"),  # NuStar Energy
        ("5020.T", "oil_refining", "JP", "Refining"),  # ENEOS Holdings
        ("5019.T", "oil_refining", "JP", "Refining"),  # Idemitsu Kosan
        ("BPCL.NS", "oil_refining", "IN", "Refining"),  # BPCL India
        ("IOC.NS", "oil_refining", "IN", "Refining"),  # Indian Oil Corp
        ("HPCL.NS", "oil_refining", "IN", "Refining"),  # HPCL India
        ("MRPL.NS", "oil_refining", "IN", "Refining"),  # MRPL India micro

        # ══════════════════════════════════════════════════════════════
        # MIDSTREAM — +13.37% α vs AMLP (63 trades)
        # Expanded with small/micro MLPs
        # ══════════════════════════════════════════════════════════════
        ("KMI", "midstream", "US", "Midstream"),
        ("WMB", "midstream", "US", "Midstream"),
        ("ET", "midstream", "US", "Midstream"),
        ("ENB", "midstream", "CA", "Midstream"),
        ("TRP", "midstream", "CA", "Midstream"),
        ("PBA", "midstream", "CA", "Midstream"),
        ("KEY.TO", "midstream", "CA", "Midstream"),
        ("ALA.TO", "midstream", "CA", "Midstream"),
        # NEW small/micro midstream
        ("AM", "midstream", "US", "Midstream"),  # Antero Midstream small
        ("CCLP", "midstream", "US", "Midstream"),  # CSI Compressco micro
        ("HESM", "midstream", "US", "Midstream"),  # Hess Midstream
        ("DTM", "midstream", "US", "Midstream"),  # DT Midstream
        ("KNTK", "midstream", "US", "Midstream"),  # Kinetik Holdings small
        ("MPLX", "midstream", "US", "Midstream"),  # MPLX LP
        ("OKE", "midstream", "US", "Midstream"),  # Oneok
        ("TRGP", "midstream", "US", "Midstream"),  # Targa Resources
        ("WES", "midstream", "US", "Midstream"),  # Western Midstream
        ("CEQP", "midstream", "US", "Midstream"),  # Crestwood Equity
        ("GEL", "midstream", "US", "Midstream"),  # Genesis Energy micro
        ("SMLP", "midstream", "US", "Midstream"),  # Summit Midstream micro
        ("SPH.TO", "midstream", "CA", "Midstream"),  # Sustain Pipeline micro
        ("PPL.TO", "midstream", "CA", "Midstream"),  # Pembina Pipeline
        ("IPL.TO", "midstream", "CA", "Midstream"),  # Inter Pipeline

        # ══════════════════════════════════════════════════════════════
        # LITHIUM — +51.45% α vs LIT (57 trades, 6 tickers)
        # MASSIVELY expanded with micro/small caps
        # ══════════════════════════════════════════════════════════════
        ("SQM", "lithium", "CL", "Lithium"),
        ("ALB", "lithium", "US", "Lithium"),
        ("SLI", "lithium", "US", "Lithium"),  # Standard Lithium micro
        ("LAC", "lithium", "CA", "Lithium"),  # Lithium Americas
        ("MIN.AX", "lithium", "AU", "Lithium"),
        ("IGO.AX", "lithium", "AU", "Lithium"),
        ("PLS.AX", "lithium", "AU", "Lithium"),
        ("LTR.AX", "lithium", "AU", "Lithium"),
        ("ALTM", "lithium", "US", "Lithium"),
        # NEW micro/small lithium
        ("LTHM", "lithium", "US", "Lithium"),  # Livent / Arcadium Lithium
        ("IONR", "lithium", "US", "Lithium"),  # ioneer Ltd micro
        ("LI.V", "lithium", "CA", "Lithium"),  # American Lithium micro (TSXV)
        ("CRE.V", "lithium", "CA", "Lithium"),  # Critical Elements micro (TSXV)
        ("PLL", "lithium", "CA", "Lithium"),  # Piedmont Lithium micro
        ("SGML", "lithium", "CA", "Lithium"),  # Sigma Lithium micro
        ("GNENF", "lithium", "AU", "Lithium"),  # Galan Lithium micro
        ("AKE.AX", "lithium", "AU", "Lithium"),  # Allkem small
        ("LKE.AX", "lithium", "AU", "Lithium"),  # Lake Resources micro
        ("CXO.AX", "lithium", "AU", "Lithium"),  # Core Lithium micro
        ("GL1.AX", "lithium", "AU", "Lithium"),  # Global Lithium micro
        ("LRS.AX", "lithium", "AU", "Lithium"),  # Latin Resources micro
        ("AGY.AX", "lithium", "AU", "Lithium"),  # Argosy Minerals micro
        ("ESS.AX", "lithium", "AU", "Lithium"),  # Essential Metals micro
        ("FFX.AX", "lithium", "AU", "Lithium"),  # FireFinch micro

        # ══════════════════════════════════════════════════════════════
        # RARE EARTHS — +57.06% α vs REMX (20 trades, 2 tickers)
        # Need WAY more tickers to validate this isn't overfitting
        # ══════════════════════════════════════════════════════════════
        ("MP", "rare_earths", "US", "Rare Earths"),
        ("LYC.AX", "rare_earths", "AU", "Rare Earths"),  # Lynas
        ("URNM", "rare_earths", "US", "Rare Earths ETF"),
        ("REE", "rare_earths", "US", "Rare Earths"),
        ("UCORE", "rare_earths", "CA", "Rare Earths"),
        ("TMRC", "rare_earths", "US", "Rare Earths"),
        # NEW micro/small rare earths & critical minerals
        ("HREE", "rare_earths", "US", "Rare Earths"),  # UCore (US listing)
        ("REEMF", "rare_earths", "CA", "Rare Earths"),  # Search Minerals micro
        ("VNCE", "rare_earths", "US", "Rare Earths"),  # Vince Holding (critical minerals)
        ("NIOBF", "rare_earths", "CA", "Rare Earths"),  # NioCorp Developments
        ("NB", "rare_earths", "CA", "Rare Earths"),  # NioBay Metals micro
        ("ARR.AX", "rare_earths", "AU", "Rare Earths"),  # Arafura Rare Earths micro
        ("ILU.AX", "rare_earths", "AU", "Rare Earths"),  # Iluka Resources (mineral sands + RE)
        ("VML.AX", "rare_earths", "AU", "Rare Earths"),  # Vital Metals micro
        ("ASM.AX", "rare_earths", "AU", "Rare Earths"),  # Australian Strategic Materials
        ("RIO.AX", "rare_earths", "AU", "Diversified Mining"),  # Rio Tinto (RE exposure)
        ("HAS.AX", "rare_earths", "AU", "Rare Earths"),  # Hastings Technology Metals micro
        ("GGG.AX", "rare_earths", "AU", "Rare Earths"),  # Greenland Minerals micro
        ("NEO.TO", "rare_earths", "CA", "Rare Earths"),  # Neo Performance Materials
        ("DEFN.V", "rare_earths", "CA", "Rare Earths"),  # Defense Metals micro
        ("USA.V", "rare_earths", "CA", "Rare Earths"),  # Americas Gold & Silver

        # ══════════════════════════════════════════════════════════════
        # PRECIOUS METALS — kept smaller (Sharpe 2.8, -2% vs GDX)
        # Focus on micro/small with highest alpha-per-trade
        # ══════════════════════════════════════════════════════════════

        # ── Gold Miners (kept top performers + added micro/small) ──
        ("GOLD", "precious_metals", "CA", "Gold"),
        ("AEM", "precious_metals", "CA", "Gold"),
        ("NEM", "precious_metals", "US", "Gold"),
        ("KGC", "precious_metals", "CA", "Gold"),
        ("AU", "precious_metals", "ZA", "Gold"),
        ("WPM", "precious_metals", "CA", "Royalty"),
        ("FNV", "precious_metals", "CA", "Royalty"),
        ("NGD", "precious_metals", "CA", "Gold"),
        ("OR", "precious_metals", "CA", "Gold"),
        ("BTG", "precious_metals", "CA", "Gold"),
        ("EQX", "precious_metals", "CA", "Gold"),
        ("SSRM", "precious_metals", "CA", "Gold"),
        ("ARIS", "precious_metals", "CA", "Gold"),
        ("IAG", "precious_metals", "CA", "Gold"),
        ("OGC.TO", "precious_metals", "CA", "Gold"),
        ("CG.TO", "precious_metals", "CA", "Gold"),
        ("TXG.TO", "precious_metals", "CA", "Gold"),
        ("NST.AX", "precious_metals", "AU", "Gold"),
        ("EVN.AX", "precious_metals", "AU", "Gold"),
        ("WAF.AX", "precious_metals", "AU", "Gold"),
        ("HMY", "precious_metals", "ZA", "Gold"),
        ("GFI", "precious_metals", "ZA", "Gold"),
        ("SBSW", "precious_metals", "ZA", "PGM"),
        # NEW micro/small gold miners
        ("GROY", "precious_metals", "US", "Royalty"),  # Gold Royalty Corp micro
        ("ELGD", "precious_metals", "US", "Gold"),  # Eldorado Gold
        ("TRX", "precious_metals", "US", "Gold"),  # Tanzania Royalty micro
        ("SAND", "precious_metals", "US", "Royalty"),  # Sandstorm Gold micro
        ("TORQ.V", "precious_metals", "CA", "Gold"),  # Torq Resources micro
        ("LGD.AX", "precious_metals", "AU", "Gold"),  # Legend Mining micro
        ("DEG.AX", "precious_metals", "AU", "Gold"),  # De Grey Mining small
        ("GOR.AX", "precious_metals", "AU", "Gold"),  # Gold Road Resources small
        ("RMS.AX", "precious_metals", "AU", "Gold"),  # Ramelius Resources small

        # ── Silver Miners (kept) ──
        ("HL", "precious_metals", "US", "Silver"),
        ("PAAS", "precious_metals", "CA", "Silver"),
        ("AG", "precious_metals", "CA", "Silver"),
        ("MAG", "precious_metals", "CA", "Silver"),
        ("FSM", "precious_metals", "CA", "Silver"),
        ("SVM", "precious_metals", "CA", "Silver"),
        ("CDE", "precious_metals", "US", "Silver"),
        ("EXK", "precious_metals", "US", "Silver"),
        ("HOC.L", "precious_metals", "UK", "Silver"),
        ("FRES.L", "precious_metals", "UK", "Silver"),

        # ══════════════════════════════════════════════════════════════
        # OIL SERVICES — kept smaller (0% vs OIH, borderline)
        # Only top performers, focus micro/small
        # ══════════════════════════════════════════════════════════════
        ("SLB", "oil_services", "US", "Oil Services"),
        ("HAL", "oil_services", "US", "Oil Services"),
        ("NOV", "oil_services", "US", "Oil Services"),
        ("FTI", "oil_services", "US", "Oil Services"),
        ("WFRD", "oil_services", "US", "Oil Services"),
        ("RIG", "oil_services", "US", "Offshore Drilling"),
        ("NE", "oil_services", "US", "Offshore Drilling"),
        ("VAL", "oil_services", "US", "Offshore Drilling"),
        # NEW micro/small oil services
        ("PUMP", "oil_services", "US", "Pressure Pumping"),  # ProPetro micro
        ("AROC", "oil_services", "US", "Compression"),  # Archrock small
        ("XPRO", "oil_services", "US", "Oil Services"),  # Expro Group micro
        ("WTTR", "oil_services", "US", "Water Services"),  # Select Water micro
        ("NINE", "oil_services", "US", "Drilling"),  # Nine Energy micro
        ("TUSK", "oil_services", "US", "Oil Services"),  # Mammoth Energy micro
        ("DEN.OL", "oil_services", "NO", "Oil Services"),  # Dof Subsea micro
        ("AKSO.OL", "oil_services", "NO", "Oil Services"),  # Aker Solutions

        # ══════════════════════════════════════════════════════════════
        # URANIUM — kept for testing (was -16% vs URA, but with
        # new supply/demand contracts may improve)
        # Focus on micro/small caps only
        # ══════════════════════════════════════════════════════════════
        ("UUUU", "uranium", "US", "Uranium"),
        ("URG", "uranium", "US", "Uranium"),
        ("UEC", "uranium", "US", "Uranium"),
        ("DNN", "uranium", "CA", "Uranium"),
        ("NXE", "uranium", "CA", "Uranium"),
        ("CCJ", "uranium", "CA", "Uranium"),
        ("PDN.AX", "uranium", "AU", "Uranium"),
        ("BOE.AX", "uranium", "AU", "Uranium"),
        ("LOT.AX", "uranium", "AU", "Uranium"),
        ("PEN.AX", "uranium", "AU", "Uranium"),
        ("GLO", "uranium", "CA", "Uranium"),
        ("EU", "uranium", "CA", "Uranium"),
        # NEW micro uranium
        ("FIND.V", "uranium", "CA", "Uranium"),  # Fission 3.0 micro
        ("FCU.TO", "uranium", "CA", "Uranium"),  # Fission Uranium micro
        ("ELVT.V", "uranium", "CA", "Uranium"),  # Elevation Gold micro
        ("NXE.TO", "uranium", "CA", "Uranium"),  # NexGen (TSX listing)
        ("STND.V", "uranium", "CA", "Uranium"),  # Standard Uranium micro
        ("FSY.TO", "uranium", "CA", "Uranium"),  # Forsys Metals micro
        ("AGE.AX", "uranium", "AU", "Uranium"),  # Alligator Energy micro
        ("BMN.AX", "uranium", "AU", "Uranium"),  # Bannerman Energy micro
        ("DYL.AX", "uranium", "AU", "Uranium"),  # Deep Yellow micro
        ("MEY.AX", "uranium", "AU", "Uranium"),  # Marenica Energy micro
    ]
)


def global_universe_tickers() -> list[str]:
    return [row.ticker for row in GLOBAL_COMMODITY_UNIVERSE]


def global_universe_by_ticker() -> dict[str, UniverseTicker]:
    return {row.ticker: row for row in GLOBAL_COMMODITY_UNIVERSE}

