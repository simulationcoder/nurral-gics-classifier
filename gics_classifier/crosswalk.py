"""Evidence -> GICS sub-industry crosswalk: rules, no training data.

Three kinds of evidence, in order of trust:

1. A Yahoo Finance industry (the 145 names yfinance/yahooquery report). Current and
   close to GICS granularity, but some industries fan out over several sub-industries.
2. A SEC SIC code (the 4-digit code on EDGAR filings). Authoritative but stale (a
   company that came public through a SPAC keeps 6770 for years) and coarse (2834
   covers pharma and biotech alike).
3. Words in the company's name, description and SIC description. Weak; used to pick
   between a Yahoo industry's candidates, or when nothing else exists.

Each map entry is ``(default_code, refinements)``; refinements is an ordered list of
``(regex, code)`` tried against the lower-cased name + SIC description + description,
first match wins, else the default. Shell companies (SPACs, blank checks) get no
placement: the GICS methodology assigns them none. Names that are plainly notes,
preferreds, depositary shares or funds are excluded too.

This module is data plus one ``classify()`` function so the mapping can be read and
reviewed as a table. Keys must be exact and unique: tests/test_crosswalk.py parses the
file with ``ast`` and fails on a duplicate key in any dict literal, because a duplicate
silently replaces the earlier entry and its refinement rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SHELL = "SHELL"  # sentinel: GICS not assigned (methodology: shell companies)

# Refinements that must say what the company DOES, not what its text mentions in passing.
# A refrigerant reclaimer that "sells industrial gas" in a list of services is not an
# industrial-gas producer, and a wristband whose blurb lists the smartphones it works with
# does not make smartphones. Both happened; both patterns below are anchored.
_INDUSTRIAL_GAS = (r"\bindustrial gas(es)? (suppliers?|producers?|company|business|production)\b"
                   r"|\b(largest|leading|major) (suppliers?|producers?) of (industrial|atmospheric) gas"
                   r"|\b(produces?|producer|supplier) of (industrial|atmospheric) gas")
_SELLS_DEVICES = (r"\b(sells?|makes?|manufactures?|designs?|produces?|maker|manufacturer) [^.]{0,80}"
                  r"\b(smartphones?|mobile phones?|personal computers?|laptops?|notebooks?|tablets?|pcs?)\b"
                  r"|\b(iphone|ipad|macbook)\b[^.]{0,60}\b(accounts? for|sales|revenue)")

# --------------------------------------------------------------------------------------
# Yahoo industry -> GICS sub-industry
# --------------------------------------------------------------------------------------
YAHOO_TO_GICS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # Basic Materials
    "Agricultural Inputs": ("15101030", []),
    "Aluminum": ("15104010", []),
    "Building Materials": ("15102010", [(r"\b(lumber|wood|timber)\b", "15105010"),
                                        (r"\b(insulation|roofing|siding|windows?|doors?|flooring)\b", "20102010")]),
    "Chemicals": ("15101010", [(_INDUSTRIAL_GAS, "15101040"),
                               (r"\b(fertili[sz]er|potash|crop)\b", "15101030"),
                               (r"\bdiversified chemical", "15101020"),
                               (r"\b(specialty|speciality|coatings?|adhesive|catalyst)\b", "15101050")]),
    "Coking Coal": ("15104050", []),
    "Copper": ("15104025", []),
    "Gold": ("15104030", [(r"\broyalt(y|ies)|streaming\b", "15104030")]),
    "Lumber & Wood Production": ("15105010", []),
    "Other Industrial Metals & Mining": ("15104020", [(r"\buranium\b", "10102050"),
                                                      (r"\biron ore\b", "15104050"),
                                                      (r"\bbauxite\b", "15104010"),
                                                      (r"\b(platinum|palladium|pgm|rare earth|diamond)\b", "15104040")]),
    "Other Precious Metals & Mining": ("15104040", [(r"\bgold\b", "15104030"), (r"\bsilver\b", "15104045")]),
    "Paper & Paper Products": ("15105020", [(r"\b(packaging|corrugated|containerboard)\b", "15103020")]),
    "Silver": ("15104045", []),
    "Specialty Chemicals": ("15101050", [(_INDUSTRIAL_GAS, "15101040")]),
    "Steel": ("15104050", []),
    # Communication Services
    "Advertising Agencies": ("50201010", []),
    "Broadcasting": ("50201020", [(r"\b(cable|satellite tv|pay television)\b", "50201030")]),
    "Electronic Gaming & Multimedia": ("50202020", []),
    "Entertainment": ("50202010", [(r"\b(theme park|amusement park|fitness|ski resort|golf)\b", "25301030")]),
    "Internet Content & Information": ("50203010", []),
    "Publishing": ("50201040", []),
    "Telecom Services": ("50101020", [(r"\b(wireless|mobile|cellular|mvno)\b", "50102010"),
                                      (r"\b(cable|broadband|pay tv|pay-tv)\b", "50201030"),
                                      (r"\b(satellite|fiber|fibre|voip|cloud communications|ucaas|cpaas|tower)\b", "50101010")]),
    # Consumer Cyclical
    "Apparel Manufacturing": ("25203010", []),
    "Apparel Retail": ("25504010", []),
    "Auto & Truck Dealerships": ("25504050", [(r"\b(truck|commercial vehicle|heavy[- ]duty)\b", "20107010")]),
    "Auto Manufacturers": ("25102010", [(r"\b(motorcycle|scooter|powersports)\b", "25102020"),
                                        (r"\b(truck|bus|commercial vehicle)\b", "20106010")]),
    "Auto Parts": ("25101010", [(r"\btires?\b", "25101020")]),
    "Department Stores": ("25503030", []),
    "Footwear & Accessories": ("25203020", [(r"\b(handbag|accessor|jewel|watch|luxury|eyewear)", "25203010")]),
    "Furnishings, Fixtures & Appliances": ("25201020", [(r"\b(appliance|refrigerat|dishwasher|washer)", "25201040"),
                                                        (r"\b(cookware|housewares|kitchenware|cutlery|tableware)\b", "25201050"),
                                                        (r"\b(office furniture|workplace)\b", "20201060"),
                                                        (r"\b(cabinet|plumbing|faucet|window|door|flooring|building products)\b", "20102010")]),
    "Gambling": ("25301010", []),
    "Home Improvement Retail": ("25504030", []),
    "Internet Retail": ("25503030", [(r"\b(pet|chewy)\b", "25504040"),
                                     (r"\b(apparel|fashion|clothing|footwear)\b", "25504010"),
                                     (r"\b(used (cars|vehicles)|auto(motive)? (retail|marketplace)|carvana|vroom)\b", "25504050"),
                                     (r"\b(furniture|home furnishing|home goods|wayfair)\b", "25504060"),
                                     (r"\b(grocer|food delivery|meal kit)\b", "30101040")]),
    "Leisure": ("25202010", [(r"\b(fitness cent|gym|health club|theme park|amusement park|ski|golf course|stadium|bowling|trampoline)\b", "25301030"),
                             (r"\b(cruise)\b", "25301020"),
                             (r"\b(toy|game|bicycle|boat|firearm|ammunition|golf (club|equipment)|sporting goods)\b", "25202010")]),
    "Lodging": ("25301020", [(r"\bcasino\b", "25301010")]),
    "Luxury Goods": ("25203010", []),
    "Packaging & Containers": ("15103010", [(r"\b(paper|corrugated|containerboard|flexible|film|label|plastic packaging|packaging materials?)\b", "15103020")]),
    "Personal Services": ("25302020", [(r"\b(education|school|training|tutoring)\b", "25302010"),
                                       (r"\b(staffing|employment)\b", "20202010")]),
    "Recreational Vehicles": ("25202010", [(r"\b(motorcycle|powersports|atv|side-by-side)\b", "25102020")]),
    "Residential Construction": ("25201030", []),
    "Resorts & Casinos": ("25301010", [(r"\b(resort|hotel|lodge)\b(?![^.]*casino)", "25301020")]),
    "Restaurants": ("25301040", []),
    "Specialty Retail": ("25504040", [(r"\b(electronics|computer|video game|game ?stop|best buy)\b", "25504020"),
                                      (r"\b(furniture|mattress|home furnishing|home d[eé]cor|bed bath)\b", "25504060"),
                                      (r"\b(auto parts|automotive (parts|aftermarket)|tires?|car wash)\b", "25504050"),
                                      (r"\b(home improvement|hardware store|lumber|building materials)\b", "25504030"),
                                      (r"\b(pharmac|drugstore|drug store)\b", "30101010"),
                                      (r"\b(dollar store|variety store|general merchandise)\b", "30101040"),
                                      (r"\b(apparel|clothing|fashion)\b", "25504010")]),
    "Textile Manufacturing": ("25203030", []),
    "Travel Services": ("25301020", [(r"\b(timeshare|vacation ownership)\b", "25301020"),
                                     (r"\b(theme park|ski|attraction)\b", "25301030")]),
    # Consumer Defensive
    "Beverages - Brewers": ("30201010", []),
    "Beverages - Non-Alcoholic": ("30201030", []),
    "Beverages - Wineries & Distilleries": ("30201020", []),
    "Confectioners": ("30202030", []),
    "Discount Stores": ("30101040", [(r"\b(closeout|off-price|five below|ollie)\b", "25503030")]),
    "Education & Training Services": ("25302010", []),
    "Farm Products": ("30202010", [(r"\b(packaged|processed|branded|snack)\b", "30202030")]),
    "Food Distribution": ("30101020", []),
    "Grocery Stores": ("30101030", [(r"\b(convenience store|gas station|fuel)\b", "30101030")]),
    "Household & Personal Products": ("30301010", [(r"\b(cosmetic|beauty|skin ?care|fragrance|personal care|hair care|oral care|grooming|razor|diaper|feminine)", "30302010")]),
    "Packaged Foods": ("30202030", []),
    "Tobacco": ("30203010", []),
    # Energy
    "Oil & Gas Drilling": ("10101010", []),
    "Oil & Gas E&P": ("10102020", []),
    "Oil & Gas Equipment & Services": ("10101020", [(r"\b(drilling contractor|owns and operates .{0,20}rigs|offshore driller)\b", "10101010")]),
    "Oil & Gas Integrated": ("10102010", []),
    "Oil & Gas Midstream": ("10102040", []),
    "Oil & Gas Refining & Marketing": ("10102030", []),
    "Thermal Coal": ("10102050", [(r"\b(metallurgical|met coal|coking)\b", "15104050")]),
    "Uranium": ("10102050", []),
    # Financial Services
    "Asset Management": ("40203010", [(r"\b(private equity|alternative asset|investment bank)\b", "40203010")]),
    "Banks - Diversified": ("40101010", []),
    "Banks - Regional": ("40101015", []),
    "Capital Markets": ("40203020", [(r"\b(stock exchange|securities exchange|clearing house|index provider|market data|cryptocurrency exchange|crypto(currency)? (platform|exchange))\b", "40203040"),
                                     (r"\b(asset management|wealth management|investment advis)", "40203010"),
                                     (r"\b(diversified capital markets)\b", "40203030")]),
    "Credit Services": ("40202010", [(r"\b(payment|card network|merchant acquir|money transfer|remittance|digital wallet|processor)", "40201060"),
                                     (r"\b(equipment (finance|leasing)|commercial (lending|finance|loans)|factoring|small business (lending|loans)|trade finance|business development company|bdc)\b", "40201040"),
                                     (r"\b(mortgage|home loans?)\b", "40201050"),
                                     (r"\b(student loan|auto loan|consumer (loan|credit|finance)|credit card|personal loan|buy now|pawn|payday|installment)", "40202010")]),
    "Financial Conglomerates": ("40201020", []),
    "Financial Data & Stock Exchanges": ("40203040", []),
    "Insurance - Diversified": ("40301030", []),
    "Insurance - Life": ("40301020", []),
    "Insurance - Property & Casualty": ("40301040", []),
    "Insurance - Reinsurance": ("40301050", []),
    "Insurance - Specialty": ("40301040", [(r"\b(health|dental|disability|supplemental|life)\b", "40301020"),
                                           (r"\breinsur", "40301050")]),
    "Insurance Brokers": ("40301010", []),
    "Mortgage Finance": ("40201050", []),
    "Shell Companies": (SHELL, []),
    # Healthcare
    "Biotechnology": ("35201010", []),
    "Diagnostics & Research": ("35203010", [(r"\b(clinical laborator|laboratory services|lab testing|reference laborator|diagnostic (testing )?services|testing services)\b", "35102015"),
                                            (r"\b(contract research|cro\b|clinical trial services|preclinical)", "35203010"),
                                            (r"\b(diagnostic (test|kit|assay|device|instrument|system)|point-of-care|in vitro diagnostic)", "35101010")]),
    "Drug Manufacturers - General": ("35202010", []),
    "Drug Manufacturers - Specialty & Generic": ("35202010", [(r"\b(cannabis|marijuana|hemp|cbd)\b", "35202010")]),
    "Health Information Services": ("35103010", [(r"\b(telehealth|virtual care|physician|clinics?)\b", "35102015")]),
    "Healthcare Plans": ("35102030", []),
    "Medical Care Facilities": ("35102020", [(r"\b(home health|hospice|dialysis|telehealth|physician (practice|group)|staffing|ambulance|pharmacy benefit|outsourced|revenue cycle|urgent care|physical therapy|primary care)", "35102015")]),
    "Medical Devices": ("35101010", [(r"\b(disposable|consumable|contact lens|glove|syringe|wound care|dental (supplies|consumables))", "35101020")]),
    "Medical Distribution": ("35102010", []),
    "Medical Instruments & Supplies": ("35101020", [(r"\b(surgical (robot|system)|robotic|imaging system|monitor(ing)? (system|device)|infusion pump|insulin pump|continuous glucose|implant|laser|endoscop|capital equipment|ventilator|defibrillator)", "35101010")]),
    "Pharmaceutical Retailers": ("30101010", []),
    # Industrials
    "Aerospace & Defense": ("20101010", []),
    "Airlines": ("20302010", [(r"\b(cargo|freight)\b", "20301010")]),
    "Airports & Air Services": ("20305010", [(r"\b(leas(e|ing) (of )?aircraft|aircraft leasing|aircraft lessor)\b", "20107010"),
                                             (r"\b(air cargo|freight|logistics)\b", "20301010"),
                                             (r"\b(charter|private (jet|aviation)|helicopter|fractional)\b", "20302010"),
                                             (r"\b(mro|maintenance, repair|aircraft parts|aerostructure)\b", "20101010")]),
    "Building Products & Equipment": ("20102010", []),
    "Business Equipment & Supplies": ("20201060", []),
    "Conglomerates": ("20105010", [(r"\b(holding company|investment holding|diversified holding|holdings? with (interests|investments))", "40201030")]),
    "Consulting Services": ("20202020", [(r"\b(it consulting|technology consulting|digital transformation|systems integrat)", "45102010")]),
    "Consumer Electronics": ("25201010", [(_SELLS_DEVICES, "45202030")]),
    "Electrical Equipment & Parts": ("20104010", [(r"\b(turbine|heavy electrical|power generation equipment|transformer|nuclear reactor|grid-scale|fuel cell)\b", "20104020")]),
    "Engineering & Construction": ("20103010", []),
    "Farm & Heavy Construction Machinery": ("20106010", [(r"\b(agricultur|farm|tractor|irrigation|harvest)", "20106015")]),
    "Industrial Distribution": ("20107010", []),
    "Infrastructure Operations": ("20305020", [(r"\b(port|terminal|marine)\b", "20305030"), (r"\bairport", "20305010")]),
    "Integrated Freight & Logistics": ("20301010", [(r"\b(trucking|truckload|less-than-truckload|ltl\b)", "20304030")]),
    "Marine Shipping": ("20303010", []),
    "Metal Fabrication": ("20106020", [(r"\b(structural steel|building products|hvac|roofing|metal buildings)\b", "20102010")]),
    "Pollution & Treatment Controls": ("20106020", [(r"\b(waste|remediation|environmental services|recycling)\b", "20201050"),
                                                    (r"\b(water (treatment|technolog|purification|filtration))", "20106020")]),
    "Railroads": ("20304010", [(r"\b(railcar (leasing|manufactur)|rail equipment|locomotive)\b", "20106010")]),
    "Rental & Leasing Services": ("20107010", [(r"\b(car rental|vehicle rental|rent-a-car|ride ?sharing|hertz|avis)\b", "20304040"),
                                               (r"\b(equipment finance|lease financing|leasing institution)\b", "40201040"),
                                               (r"\b(rent-to-own|lease-to-own|consumer leasing)\b", "25504040"),
                                               (r"\b(uniform|linen|textile rental)\b", "20201070")]),
    "Security & Protection Services": ("20201080", [(r"\b(cybersecurity|cyber security|identity)\b", "45103020"),
                                                    (r"\b(detection equipment|screening (equipment|systems)|surveillance equipment|sensor)", "45203010"),
                                                    (r"\b(body armor|defense|military)\b", "20101010")]),
    "Specialty Business Services": ("20201070", [(r"\b(data processing|business process outsourc|bpo\b|call cent|customer experience|back-office)", "20202030"),
                                                 (r"\b(payment|transaction processing)\b", "40201060"),
                                                 (r"\b(commercial printing|print(ing)? services)\b", "20201010"),
                                                 (r"\b(research|consulting|advisory|market intelligence)\b", "20202020"),
                                                 (r"\b(staffing|recruit|workforce)\b", "20202010"),
                                                 (r"\b(facilit(y|ies) (services|management)|janitorial|environmental)\b", "20201050"),
                                                 (r"\b(security (services|guard)|alarm)\b", "20201080"),
                                                 (r"\b(office (supplies|products))\b", "20201060")]),
    "Specialty Industrial Machinery": ("20106020", [(r"\b(hvac|heating, ventilat|air conditioning|air-conditioning|climate control|building products)\b", "20102010"),
                                                    (r"\b(gas turbine|steam turbine|wind turbine|generator sets?|power generation equipment|transformers?)\b", "20104020"),
                                                    (r"\b(agricultur|farm equipment|irrigation)", "20106015"),
                                                    (r"\b(construction (equipment|machinery)|mining (equipment|machinery)|crane|forklift|excavator|railcar|truck bodies)\b", "20106010"),
                                                    (r"\b(semiconductor (equipment|capital)|wafer)\b", "45301010"),
                                                    (r"\b(aerospace|defense)\b", "20101010")]),
    "Staffing & Employment Services": ("20202010", []),
    "Tools & Accessories": ("20106020", []),
    "Trucking": ("20304030", []),
    "Waste Management": ("20201050", []),
    # Real Estate
    "REIT - Diversified": ("60101010", []),
    "REIT - Healthcare Facilities": ("60105010", []),
    "REIT - Hotel & Motel": ("60103010", []),
    "REIT - Industrial": ("60102510", []),
    "REIT - Mortgage": ("40204010", []),
    "REIT - Office": ("60104010", []),
    "REIT - Residential": ("60106010", [(r"\b(single[- ]family|manufactured (home|housing)|mobile home)", "60106020")]),
    "REIT - Retail": ("60107010", []),
    "REIT - Specialty": ("60108010", [(r"\bself[- ]storage\b", "60108020"),
                                      (r"\b(tower|wireless infrastructure|cell site)", "60108030"),
                                      (r"\btimber", "60108040"),
                                      (r"\bdata cent", "60108050"),
                                      (r"\b(net lease|net-lease|single[- ]tenant)\b", "60107010")]),
    "Real Estate - Development": ("60201030", []),
    "Real Estate - Diversified": ("60201010", []),
    "Real Estate Services": ("60201040", [(r"\b(owns and operates|owner and operator|owns, operates|landlord|portfolio of (properties|real estate)|leasing of its)\b", "60201020")]),
    # Technology
    "Communication Equipment": ("45201020", []),
    "Computer Hardware": ("45202030", [(r"\b(barcode|point[- ]of[- ]sale|scanner|rugged|display)\b", "45203010")]),
    "Electronic Components": ("45203015", [(r"\b(contract manufactur|electronic manufacturing services|ems\b|printed circuit board assembl)", "45203020")]),
    "Electronics & Computer Distribution": ("45203030", []),
    "Information Technology Services": ("45102010", [(r"\b(payment|transaction processing|merchant)\b", "40201060"),
                                                     (r"\b(business process outsourc|bpo\b|call cent|customer experience|data processing|back-office)", "20202030"),
                                                     (r"\b(data cent(er|re)|cloud infrastructure|web hosting|hosting|domain|content delivery|cdn\b|colocation)", "45102030"),
                                                     (r"\b(bitcoin|crypto(currency)? mining|digital asset mining)", "45103010")]),
    "Scientific & Technical Instruments": ("45203010", [(r"\b(life science|laboratory|lab\b|analytical instrument|mass spectrom|chromatograph|bioprocess)", "35203010"),
                                                        (r"\b(medical|surgical|patient)\b", "35101010")]),
    "Semiconductor Equipment & Materials": ("45301010", []),
    "Semiconductors": ("45301020", []),
    "Software - Application": ("45103010", [(r"\b(payment|payments platform|digital wallet)\b", "40201060"),
                                            (r"\b(bitcoin|crypto(currency)? mining)", "45103010")]),
    "Software - Infrastructure": ("45103020", [(r"\b(payment|payments platform|digital wallet|merchant)\b", "40201060"),
                                               (r"\b(data cent(er|re)|web hosting|hosting provider|domain (name|registr)|content delivery|cdn\b|colocation|cloud infrastructure provider)", "45102030"),
                                               (r"\b(bitcoin|crypto(currency)? mining|digital asset mining)", "45103010"),
                                               (r"\b(cybersecurity|cyber security|security software|identity|endpoint|firewall|operating system|database|devops|developer tools|infrastructure software|middleware|storage software)", "45103020"),
                                               (r"\b(e-?commerce platform|marketplace|crm|erp|analytics|collaboration|design software|marketing (software|platform)|human capital|accounting software)", "45103010")]),
    "Solar": ("45301020", [(r"\b(install|residential solar|rooftop|solar (energy )?services|solar financing|home solar)", "20104010"),
                           (r"\b(tracker|racking|mounting|inverter|microinverter|optimizer|balance of system|energy storage system)", "20104010"),
                           (r"\b(independent power|power plant|yieldco|owns and operates .{0,30}(solar|renewable)|sells electricity|power purchase)", "55105020"),
                           (r"\b(polysilicon|wafer)\b", "45301010")]),
    # Utilities
    "Utilities - Diversified": ("55103010", []),
    "Utilities - Independent Power Producers": ("55105010", []),
    "Utilities - Regulated Electric": ("55101010", []),
    "Utilities - Regulated Gas": ("55102010", [(r"\b(propane|heating oil|fuel distribut)", "10102040")]),
    "Utilities - Regulated Water": ("55104010", []),
    "Utilities - Renewable": ("55105020", [(r"\b(renewable natural gas|biogas|hydrogen (production|fuel)|fuel cell)\b", "10102050")]),
}
# Keys are looked up verbatim against the Yahoo node name, so a key with stray
# whitespace is a dead entry -- and once it was worse: "Consumer Electronics " replaced
# the real entry when the map was normalised, erased the iPhone rule and put Apple in
# Consumer Discretionary. Refuse such keys outright (a raise, not an assert, so -O cannot
# strip it). Exact duplicate keys cannot be seen after the literal is built; the ast
# check in tests/test_crosswalk.py catches those in every map here.
_untrimmed = [k for k in YAHOO_TO_GICS if k != k.strip()]
if _untrimmed:
    raise ValueError(f"Yahoo industry keys with surrounding whitespace: {_untrimmed!r}")

# --------------------------------------------------------------------------------------
# SIC code -> GICS sub-industry. Exact 4-digit code first, then the longest prefix.
# --------------------------------------------------------------------------------------
_BIOTECH = (r"\b(biotech|biopharma|clinical[- ]stage|preclinical|monoclonal|antibod|gene (therapy|editing)|cell therapy|"
            r"oncology|immunotherap|rna\b|mrna|crispr|vaccine candidate|phase (1|2|3|i|ii|iii)\b|pipeline of|drug candidate|"
            r"therapeutic candidate|product candidate|platform technology)")
_REIT_REFINE = [
    (r"\bmortgage\b", "40204010"),
    (r"\b(data cent)", "60108050"),
    (r"\bself[- ]storage\b", "60108020"),
    (r"\b(tower|wireless infrastructure)\b", "60108030"),
    (r"\btimber", "60108040"),
    (r"\b(health ?care|senior (housing|living)|medical office|skilled nursing|hospital)", "60105010"),
    (r"\b(hotel|lodging|resort)\b", "60103010"),
    (r"\b(industrial|logistics|warehouse|distribution cent)", "60102510"),
    (r"\b(office)\b", "60104010"),
    (r"\b(single[- ]family|manufactured (home|housing))", "60106020"),
    (r"\b(apartment|multifamily|multi-family|residential|student housing)\b", "60106010"),
    (r"\b(shopping cent|retail|mall|net lease|net-lease|grocery-anchored|outlet)\b", "60107010"),
    (r"\b(farmland|casino|gaming|billboard|outdoor advertising|prison|correctional|cannabis|infrastructure|energy|pipeline|ground lease|parking|cold storage|land)\b", "60108010"),
    (r"\bdiversified\b", "60101010"),
]
_SOFTWARE_REFINE = [
    (r"\b(payment|payments platform|digital wallet|merchant acquir)", "40201060"),
    (r"\b(bitcoin|crypto(currency)? mining|digital asset mining)", "45103010"),
    (r"\b(data cent(er|re)|web hosting|domain (name|registr)|content delivery|cdn\b|colocation)", "45102030"),
    (r"\b(cybersecurity|cyber security|security software|operating system|database (software|platform)|devops|infrastructure software|middleware|network management|storage software|endpoint)", "45103020"),
    (r"\b(social (network|media)|search engine|online (marketplace|platform|community)|user-generated|streaming (service|platform)|digital media|content platform)\b", "50203010"),
    (r"\b(video game|mobile game|games? developer|gaming (studio|company))\b", "50202020"),
    (r"\b(electronic health|healthcare (software|it|technology)|health information|clinical (software|data)|telehealth platform)", "35103010"),
]
_BIZ_SERVICES_REFINE = [
    (r"\b(data processing|business process outsourc|bpo\b|call cent|customer experience|back-office|investor communications)", "20202030"),
    (r"\b(payment|transaction processing|merchant)\b", "40201060"),
    (r"\b(research|consulting|advisory|market intelligence|testing, inspection|certification)\b", "20202020"),
    (r"\b(staffing|recruit|workforce solutions)\b", "20202010"),
    (r"\b(marketing|advertising|digital media|media buying)\b", "50201010"),
    (r"\b(software|saas|platform|cloud)\b", "45103010"),
    (r"\b(internet|online marketplace|website|app\b)", "50203010"),
    (r"\b(health ?care|medical|patient|clinic)", "35102015"),
    (r"\b(education|school|training|learning)\b", "25302010"),
    (r"\b(logistics|freight|supply chain)\b", "20301010"),
    (r"\b(security|guard|alarm)\b", "20201080"),
    (r"\b(facilit(y|ies)|janitorial|waste|environmental)\b", "20201050"),
    (r"\b(printing)\b", "20201010"),
    (r"\b(auction)\b", "20201070"),
    (r"\b(funeral|wedding|day care|child care|home services|pest control|lawn|personal (services|care))\b", "25302020"),
    (r"\b(bank|lending|loan|credit)\b", "40201040"),
    (r"\b(real estate)\b", "60201040"),
    (r"\b(crypto|bitcoin|blockchain|digital asset)", "45103010"),
    (r"\b(entertainment|film|music|live events?|sports team)\b", "50202010"),
    (r"\b(travel|hotel|hospitality)\b", "25301020"),
    (r"\b(restaurant|food)\b", "25301040"),
    (r"\b(oil|gas|energy|petroleum)\b", "10101020"),
    (r"\b(mining|mineral|exploration)\b", "15104020"),
]
_FIN_SERVICES_REFINE = [
    (r"\b(bitcoin|crypto(currency)? mining|digital asset mining)", "45103010"),
    (r"\b(cryptocurrency exchange|crypto(currency)? (platform|exchange)|stock exchange|securities exchange|clearing|market data|index provider)\b", "40203040"),
    (r"\b(payment|money transfer|remittance|digital wallet|merchant|card issu|prepaid)", "40201060"),
    (r"\b(mortgage|home loan|real estate (lending|loans))", "40201050"),
    (r"\b(consumer (loan|lending|credit|finance)|personal loan|auto loan|student loan|credit card|installment|pawn|payday|buy now)", "40202010"),
    (r"\b(asset management|investment management|wealth management|investment advis|fund manager|private equity|alternative asset)", "40203010"),
    (r"\b(broker-dealer|brokerage|investment bank|underwrit|m&a advisory|capital markets)\b", "40203020"),
    (r"\b(insurance)\b", "40301040"),
    (r"\b(bank|bancorp)\b", "40101015"),
    (r"\b(holding company|invest(s|ment) in (a )?(diverse|variety|range|portfolio)|diversified holdings?)", "40201030"),
    (r"\b(reit|real estate investment trust)\b", "60101010"),
    (r"\b(oil|gas|mineral|royalt)", "10102020"),
    (r"\b(business development company|bdc\b|middle[- ]market (lending|loans)|direct lending)", "40203010"),
    (r"\b(equipment (finance|leasing)|commercial (lending|finance|loans)|factoring|small business (lending|loans)|trade finance|leasing)\b", "40201040"),
]
_REAL_ESTATE_REFINE = [
    (r"\b(reit|real estate investment trust)\b", "60101010"),
    (r"\b(develop(s|er|ment) of|land development|master[- ]planned|homebuild|residential communit)", "60201030"),
    (r"\b(broker|brokerage|agents?|appraisal|title|property management services|listing)\b", "60201040"),
    (r"\b(owns and operates|owner and operator|owns, operates|landlord|leas(es|ing) (its|the) propert|rental propert|portfolio of)\b", "60201020"),
]

SIC_TO_GICS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "0100": ("30202010", []), "0200": ("30202010", []), "0700": ("30202010", []), "0800": ("15105010", []), "0900": ("30202010", []),
    "1000": ("15104020", [(r"\bgold\b", "15104030"), (r"\bsilver\b", "15104045"), (r"\bcopper\b", "15104025"), (r"\buranium\b", "10102050"),
                          (r"\b(lithium|nickel|zinc|cobalt|graphite|rare earth|vanadium|tin|tungsten|antimony)\b", "15104020"),
                          (r"\b(platinum|palladium|pgm|diamond)\b", "15104040"), (r"\biron ore\b", "15104050")]),
    "1040": ("15104030", [(r"\bsilver\b(?![^.]*\bgold\b)", "15104045"), (r"\b(platinum|palladium|pgm)\b", "15104040")]),
    "1090": ("15104020", [(r"\buranium\b", "10102050"), (r"\b(platinum|palladium|pgm|diamond)\b", "15104040"), (r"\biron ore\b", "15104050")]),
    "1220": ("10102050", [(r"\b(metallurgical|met coal|coking)\b", "15104050")]),
    "1221": ("10102050", [(r"\b(metallurgical|met coal|coking)\b", "15104050")]),
    "1311": ("10102020", [(r"\b(midstream|pipeline|gathering)\b", "10102040"), (r"\bhelium\b", "15101040")]),
    "1381": ("10101010", []), "1382": ("10101020", []), "1389": ("10101020", []),
    "1400": ("15102010", [(r"\b(lithium|potash|phosphate|boron|borate)\b", "15101030"), (r"\b(sand|frac sand|proppant)\b", "10101020"),
                          (r"\b(graphite|rare earth|mineral sands|salt)\b", "15104020"), (r"\bdiamond\b", "15104040")]),
    "1520": ("25201030", []), "1531": ("25201030", []), "1540": ("20103010", []),
    "1600": ("20103010", []), "1623": ("20103010", []), "1700": ("20103010", []), "1731": ("20103010", []),
    "2000": ("30202030", []), "2011": ("30202030", []), "2013": ("30202030", []), "2015": ("30202030", []), "2020": ("30202030", []),
    "2030": ("30202030", []), "2033": ("30202030", []), "2040": ("30202030", []), "2052": ("30202030", []), "2060": ("30202030", []),
    "2070": ("30202030", [(r"\b(renewable diesel|biodiesel)\b", "10102030")]), "2090": ("30202030", [(r"\b(coffee|tea)\b", "30202030")]),
    "2080": ("30201030", [(r"\b(beer|brew)", "30201010"), (r"\b(wine|spirit|whiskey|whisky|distill|vodka|tequila)", "30201020")]),
    "2082": ("30201010", []), "2086": ("30201030", []),
    "2100": ("30203010", []), "2111": ("30203010", []),
    "2200": ("25203030", []), "2211": ("25203030", []), "2221": ("25203030", []), "2273": ("25203030", []),
    "2300": ("25203010", []), "2320": ("25203010", []), "2330": ("25203010", []),
    "2400": ("15105010", []), "2421": ("15105010", []), "2430": ("15105010", []), "2451": ("25201030", []),
    "2510": ("25201020", []), "2511": ("25201020", []), "2520": ("20201060", []), "2522": ("20201060", []), "2531": ("20201060", []),
    "2611": ("15105020", []), "2621": ("15105020", []), "2631": ("15105020", []), "2650": ("15103020", []), "2670": ("15103020", []), "2673": ("15103020", []),
    "2711": ("50201040", []), "2721": ("50201040", []), "2731": ("50201040", [(r"\b(education|learning|school)\b", "25302010")]), "2741": ("50201040", []),
    "2750": ("20201010", []), "2761": ("20201010", []), "2780": ("20201060", []),
    "2800": ("15101010", [(_INDUSTRIAL_GAS, "15101040"), (r"\b(specialty|speciality|coating|adhesive)", "15101050"), (r"\b(fertili|potash|crop)", "15101030"), (r"\bdiversified\b", "15101020")]),
    "2810": ("15101010", [(_INDUSTRIAL_GAS, "15101040"), (r"\b(lithium|battery)\b", "15101050")]),
    "2820": ("15101010", [(r"\bspecialty\b", "15101050")]), "2821": ("15101010", [(r"\bspecialty\b", "15101050")]),
    "2833": ("35202010", [(_BIOTECH, "35201010"), (r"\b(cannabis|hemp|cbd|nutraceutical|supplement|botanical)\b", "30302010")]),
    "2834": ("35202010", [(_BIOTECH, "35201010")]),
    "2835": ("35101020", [(_BIOTECH, "35201010"), (r"\b(laborator(y|ies) services|clinical laborator|testing services)\b", "35102015"),
                          (r"\b(research (tools|reagents)|life science)", "35203010"), (r"\b(instrument|system|analyzer|platform)\b", "35101010")]),
    "2836": ("35201010", []),
    "2840": ("30301010", [(r"\b(cosmetic|beauty|skin|personal care|fragrance)", "30302010")]), "2842": ("30301010", []), "2844": ("30302010", []),
    "2851": ("15101050", []), "2860": ("15101010", [(r"\b(renewable fuel|ethanol|biodiesel)\b", "10102030"), (r"\bspecialty\b", "15101050")]),
    "2870": ("15101030", []), "2890": ("15101050", []), "2891": ("15101050", []),
    "2911": ("10102030", []), "2990": ("10102030", []),
    "3011": ("25101020", []), "3021": ("25203020", []), "3050": ("20106020", []), "3060": ("20106020", [(r"\b(automotive|vehicle)\b", "25101010")]),
    "3080": ("15103020", []), "3086": ("15103020", []), "3089": ("15103020", [(r"\b(automotive|vehicle)\b", "25101010"), (r"\b(building|pipe|construction)\b", "20102010"), (r"\b(medical|surgical)\b", "35101020")]),
    "3100": ("25203010", []), "3140": ("25203020", []),
    "3211": ("20102010", []), "3221": ("15103010", []), "3231": ("20102010", []), "3241": ("15102010", []), "3260": ("25201050", []),
    "3272": ("15102010", []), "3281": ("15102010", []), "3290": ("15102010", [(r"\b(abrasive|ceramic|composite)\b", "20106020")]),
    "3310": ("15104050", []), "3312": ("15104050", []), "3317": ("15104050", [(r"\b(oil|gas|oilfield|octg)\b", "10101020")]),
    "3330": ("15104020", []), "3334": ("15104010", []), "3341": ("15104020", []), "3350": ("15104010", [(r"\bcopper\b", "15104025")]),
    "3357": ("20104010", []), "3360": ("20106020", []), "3390": ("15104050", []),
    "3411": ("15103010", []), "3412": ("15103010", []), "3420": ("20106020", []), "3430": ("20102010", []), "3433": ("20102010", []),
    "3440": ("20102010", []), "3442": ("20102010", []), "3443": ("20106020", []), "3452": ("20106020", []), "3460": ("20106020", []),
    "3470": ("20106020", []), "3480": ("20101010", [(r"\b(firearm|ammunition|sporting|hunting)\b", "25202010")]), "3490": ("20106020", []),
    "3510": ("20106020", [(r"\b(turbine|power generation|generator)\b", "20104020")]),
    "3523": ("20106015", []), "3524": ("20106015", []), "3530": ("20106010", []), "3531": ("20106010", []), "3533": ("10101020", []),
    "3537": ("20106010", []), "3540": ("20106020", []), "3541": ("20106020", []),
    "3550": ("20106020", [(r"\b(semiconductor|wafer)\b", "45301010")]), "3559": ("20106020", [(r"\b(semiconductor|wafer|photomask|lithograph)\b", "45301010"), (r"\b(3d print|additive manufactur)", "20106020")]),
    "3560": ("20106020", []), "3561": ("20106020", []), "3562": ("20106020", []), "3564": ("20106020", []), "3569": ("20106020", []),
    "3570": ("45202030", []), "3571": ("45202030", []), "3572": ("45202030", []), "3576": ("45201020", []), "3577": ("45202030", [(r"\b(barcode|point[- ]of[- ]sale|scanner)\b", "45203010")]),
    "3578": ("45203010", [(r"\b(atm|payment|point[- ]of[- ]sale)\b", "45203010")]), "3579": ("20201060", []),
    "3580": ("20106020", []), "3585": ("20102010", []), "3590": ("20106020", []),
    "3600": ("45203010", []), "3612": ("20104010", []), "3613": ("20104010", []), "3620": ("20104010", []), "3621": ("20104010", []),
    "3630": ("25201040", []), "3634": ("25201040", []), "3640": ("20104010", []), "3651": ("25201010", []),
    "3661": ("45201020", []), "3663": ("45201020", [(r"\b(satellite (operator|services)|broadcast(ing)? (services|network))\b", "50101010")]), "3669": ("45201020", []),
    "3670": ("45203015", []), "3672": ("45203020", []), "3674": ("45301020", [(r"\b(equipment|lithograph|wafer fab|deposition|etch)\b", "45301010")]),
    "3677": ("45203015", []), "3678": ("45203015", []), "3679": ("45203015", []),
    "3690": ("20104010", [(r"\b(fuel cell|hydrogen)\b", "20104020"), (r"\b(consumer|headphone|audio)\b", "25201010")]),
    "3711": ("25102010", [(r"\b(truck|bus|commercial vehicle)\b", "20106010"), (r"\b(motorcycle)\b", "25102020")]),
    "3713": ("20106010", []), "3714": ("25101010", []), "3715": ("20106010", []), "3716": ("25202010", []),
    "3720": ("20101010", []), "3721": ("20101010", []), "3724": ("20101010", []), "3728": ("20101010", []),
    "3730": ("25202010", [(r"\b(ship|naval|defense|military|navy)\b", "20101010")]), "3743": ("20106010", []), "3751": ("25102020", [(r"\bbicycle|e-bike", "25202010")]),
    "3760": ("20101010", []), "3790": ("20106010", []),
    "3812": ("20101010", []), "3821": ("35203010", []), "3822": ("20102010", []), "3823": ("45203010", []), "3824": ("45203010", []),
    "3825": ("45203010", []), "3826": ("35203010", []), "3827": ("45203010", [(r"\b(semiconductor|wafer)\b", "45301010")]), "3829": ("45203010", []),
    "3841": ("35101010", [(r"\b(disposable|consumable|syringe|glove|wound care)\b", "35101020")]), "3842": ("35101020", []), "3843": ("35101020", []),
    "3844": ("35101010", []), "3845": ("35101010", []), "3851": ("35101020", []), "3861": ("45203010", []), "3873": ("25203010", []),
    "3910": ("25203010", []), "3942": ("25202010", []), "3944": ("25202010", []), "3949": ("25202010", []), "3990": ("25201050", []),
    "4011": ("20304010", []), "4100": ("20304040", []), "4210": ("20304030", []), "4213": ("20304030", []), "4220": ("20301010", []),
    "4400": ("20303010", []), "4412": ("20303010", []), "4512": ("20302010", [(r"\b(cargo|freight)\b", "20301010")]), "4513": ("20301010", []),
    "4522": ("20302010", [(r"\b(cargo|freight)\b", "20301010"), (r"\b(aircraft leasing|lessor)\b", "20107010")]), "4581": ("20305010", []),
    "4610": ("10102040", []), "4700": ("20301010", []), "4731": ("20301010", []),
    "4812": ("50102010", []), "4813": ("50101020", [(r"\b(fiber|fibre|cloud communications|ucaas|voip)\b", "50101010")]), "4822": ("50101010", []),
    "4832": ("50201020", []), "4833": ("50201020", []), "4841": ("50201030", []),
    "4899": ("50101010", [(r"\b(satellite (imagery|data|earth)|earth observation)\b", "45203010"), (r"\b(tower)\b", "60108030"), (r"\b(streaming|content|media)\b", "50202010")]),
    "4900": ("55103010", []), "4911": ("55101010", [(r"\b(renewable|wind|solar|hydro)\b(?![^.]*regulated)", "55105020"), (r"\b(independent power|merchant)\b", "55105010")]),
    "4922": ("10102040", []), "4923": ("55102010", [(r"\b(pipeline|midstream|interstate)\b", "10102040")]), "4924": ("55102010", []),
    "4931": ("55103010", []), "4932": ("55102010", []), "4941": ("55104010", []), "4950": ("20201050", []), "4953": ("20201050", []), "4955": ("20201050", []),
    "4991": ("55105010", [(r"\b(renewable|wind|solar|hydro|geothermal)\b", "55105020")]),
    "5000": ("20107010", []), "5010": ("25501010", []), "5013": ("25501010", []), "5030": ("20107010", []), "5031": ("20107010", []),
    "5040": ("20107010", []), "5045": ("45203030", []), "5047": ("35102010", []), "5050": ("20107010", []), "5051": ("20107010", []),
    "5063": ("20107010", []), "5064": ("25501010", []), "5065": ("45203030", []), "5070": ("20107010", []), "5072": ("20107010", []),
    "5080": ("20107010", []), "5084": ("20107010", []), "5090": ("25501010", []), "5094": ("25501010", []), "5099": ("25501010", []),
    "5122": ("35102010", [(r"\b(cannabis|nutraceutical|supplement|beauty|cosmetic)\b", "30302010")]), "5130": ("25501010", []),
    "5140": ("30101020", []), "5141": ("30101020", []), "5150": ("30202010", []), "5160": ("20107010", []),
    "5171": ("10102040", []), "5172": ("10102040", [(r"\b(gas station|convenience store)\b", "25504050")]), "5180": ("30101020", []), "5190": ("30101020", []),
    "5200": ("25504030", []), "5211": ("25504030", []), "5311": ("25503030", []), "5331": ("30101040", []),
    "5400": ("30101030", []), "5411": ("30101030", []), "5412": ("30101030", []), "5500": ("25504050", []), "5531": ("25504050", []),
    "5600": ("25504010", []), "5621": ("25504010", []), "5651": ("25504010", []), "5661": ("25504010", []),
    "5700": ("25504060", []), "5712": ("25504060", []), "5731": ("25504020", []), "5734": ("25504020", []),
    "5810": ("25301040", []), "5812": ("25301040", []), "5900": ("25504040", [(r"\b(pharmac|drug)", "30101010")]), "5912": ("30101010", []),
    "5940": ("25504040", []), "5944": ("25504040", []), "5945": ("25504040", []), "5960": ("25503030", []), "5961": ("25503030", [(r"\b(pet|chewy)\b", "25504040"), (r"\b(apparel|fashion)\b", "25504010"), (r"\b(furniture|home)\b", "25504060")]),
    "5990": ("25504040", [(r"\b(pharmac|drug)", "30101010"), (r"\b(cannabis dispens)", "35202010")]),
    "6021": ("40101015", []), "6022": ("40101015", []), "6029": ("40101015", []), "6035": ("40101015", []), "6036": ("40101015", []),
    "6099": ("40201060", [(r"\b(crypto|bitcoin|digital asset)\b", "40203040"), (r"\b(bank|trust company)\b", "40101015")]),
    "6111": ("40201050", [(r"\b(agricultur|farm)\b", "40201040")]), "6141": ("40202010", []), "6153": ("40201040", []), "6159": ("40201040", []),
    "6162": ("40201050", []), "6163": ("40201050", []),
    "6199": ("40201040", _FIN_SERVICES_REFINE), "6200": ("40203020", [(r"\b(exchange|market data|clearing)\b", "40203040"), (r"\b(crypto|bitcoin|digital asset)\b", "40203040")]),
    "6211": ("40203020", [(r"\b(asset management|wealth management|investment advis)", "40203010"), (r"\b(exchange|market data|alternative trading)\b", "40203040")]),
    "6221": ("40203020", []), "6282": ("40203010", []),
    "6311": ("40301020", []), "6321": ("40301020", []), "6324": ("35102030", []), "6331": ("40301040", []), "6351": ("40301040", []), "6361": ("40301040", []),
    "6399": ("40301030", []), "6411": ("40301010", []),
    "6500": ("60201020", _REAL_ESTATE_REFINE), "6510": ("60201020", _REAL_ESTATE_REFINE), "6512": ("60201020", _REAL_ESTATE_REFINE),
    "6513": ("60201020", []), "6519": ("60201020", _REAL_ESTATE_REFINE), "6531": ("60201040", []), "6552": ("60201030", []),
    "6770": (SHELL, []),
    "6792": ("10102020", []), "6794": ("40201040", [(r"\b(franchis)", "25301040"), (r"\b(brand|licens(e|ing) (its|the|of) (brand|trademark))", "25203010"), (r"\b(patent|intellectual property|technology licens)", "45203010")]),
    "6795": ("15104020", [(r"\b(gold|precious)\b", "15104030"), (r"\b(oil|gas)\b", "10102020")]),
    "6798": ("60101010", _REIT_REFINE), "6799": ("40201030", _FIN_SERVICES_REFINE),
    "7000": ("25301020", []), "7011": ("25301020", [(r"\bcasino\b", "25301010")]), "7200": ("25302020", []),
    "7310": ("50201010", []), "7311": ("50201010", []), "7320": ("20202020", []), "7330": ("20201070", []), "7331": ("50201010", []),
    "7340": ("20201050", []), "7350": ("20107010", []), "7359": ("20107010", [(r"\b(car rental|vehicle rental)\b", "20304040")]),
    "7361": ("20202010", []), "7363": ("20202010", []),
    "7370": ("45103010", _SOFTWARE_REFINE), "7371": ("45103010", _SOFTWARE_REFINE + [(r"\b(it services|consulting|outsourc)", "45102010")]), "7372": ("45103010", _SOFTWARE_REFINE),
    "7373": ("45102010", _SOFTWARE_REFINE), "7374": ("20202030", [(r"\b(payment|transaction processing|merchant)\b", "40201060"), (r"\b(cloud|hosting|data cent|internet infrastructure|domain)", "45102030"),
                                                                  (r"\b(online|website|platform|marketplace|social|search)\b", "50203010"), (r"\b(bitcoin|crypto(currency)? mining)", "45103010"),
                                                                  (r"\b(software|saas)\b", "45103010")]),
    "7377": ("45102010", []), "7380": ("20201070", _BIZ_SERVICES_REFINE), "7381": ("20201080", []), "7389": ("20201070", _BIZ_SERVICES_REFINE),
    "7500": ("25302020", [(r"\b(car wash)\b", "25504050")]), "7510": ("20304040", []), "7600": ("20201070", []),
    "7812": ("50202010", []), "7830": ("50202010", []), "7841": ("50202010", []),
    "7900": ("25301030", [(r"\bcasino\b", "25301010"), (r"\b(entertainment|live events?|concert|media|esports|film)\b", "50202010")]),
    "7948": ("25301030", [(r"\b(casino|wagering|betting)\b", "25301010")]),
    "7990": ("25301030", [(r"\bcasino|gaming|betting|wagering\b", "25301010"), (r"\b(entertainment|live events?|media|esports)\b", "50202010")]),
    "7997": ("25301030", []),
    "8000": ("35102015", [(r"\b(hospital|surgery cent|surgical cent|nursing|rehabilitation cent|clinics?\b)", "35102020")]),
    "8011": ("35102015", [(r"\b(clinics|centers)\b", "35102020")]), "8050": ("35102020", []), "8051": ("35102020", []), "8060": ("35102020", []),
    "8062": ("35102020", []), "8071": ("35102015", [(r"\b(genomic|sequencing|research)\b", "35203010")]), "8082": ("35102015", []),
    "8090": ("35102015", [(r"\b(hospital|surgery cent|senior living|behavioral health (facilit|hospital)|rehabilitation (facilit|hospital))", "35102020")]),
    "8093": ("35102020", []), "8111": ("25302020", []), "8200": ("25302010", []), "8351": ("25302020", []),
    "8700": ("20202020", []), "8711": ("20103010", []),
    "8731": ("35203010", [(_BIOTECH, "35201010"), (r"\b(materials|battery|energy|chemical|nanotech)", "15101050"), (r"\b(software|ai\b|artificial intelligence)", "45103010")]),
    "8734": ("20202020", []), "8741": ("20202020", []), "8742": ("20202020", []), "8744": ("20201070", []),
}

SIC_PREFIX_TO_GICS: dict[str, str] = {
    "01": "30202010", "02": "30202010", "07": "30202010", "08": "15105010", "09": "30202010",
    "10": "15104020", "12": "10102050", "13": "10102020", "14": "15102010", "15": "20103010", "16": "20103010", "17": "20103010",
    "20": "30202030", "21": "30203010", "22": "25203030", "23": "25203010", "24": "15105010", "25": "25201020", "26": "15105020",
    "27": "50201040", "28": "15101010", "29": "10102030", "30": "20106020", "31": "25203010", "32": "15102010", "33": "15104050",
    "34": "20106020", "35": "20106020", "36": "45203010", "37": "20106010", "38": "45203010", "39": "25202010",
    "40": "20304010", "41": "20304040", "42": "20304030", "44": "20303010", "45": "20302010", "46": "10102040", "47": "20301010",
    "48": "50101020", "49": "55103010", "50": "20107010", "51": "30101020", "52": "25504030", "53": "25503030", "54": "30101030",
    "55": "25504050", "56": "25504010", "57": "25504060", "58": "25301040", "59": "25504040",
    "60": "40101015", "61": "40201040", "62": "40203020", "63": "40301040", "64": "40301010", "65": "60201020", "67": "40201030",
    "70": "25301020", "72": "25302020", "73": "20201070", "75": "25302020", "76": "20201070", "78": "50202010", "79": "25301030",
    "80": "35102015", "81": "25302020", "82": "25302010", "83": "25302020", "86": "25302020", "87": "20202020", "89": "20201070",
}

# Name/description rules for securities with neither a Yahoo industry nor a SIC code.
# Ordered: the first match wins. Deliberately conservative -- only words that leave
# little doubt about the sector. Everything else stays unclassified.
NAME_RULES: list[tuple[str, str]] = [
    (r"\b(bancorp|bancshares|bankshares|bank\b|banc\b|savings (bank|institution))", "40101015"),
    (r"\b(mortgage trust|mortgage reit)", "40204010"),
    (r"\b(reit\b|real estate investment trust|realty trust|properties trust|realty\b)", "60101010"),
    (r"\b(therapeutics|biotherapeutics|biosciences|biopharma|bio\b|biotech|genomics|oncology|immuno)", "35201010"),
    (r"\b(pharmaceuticals?|pharma\b)", "35202010"),
    (r"\b(uranium)\b", "10102050"),
    (r"\b(gold|silver|mining|minerals|mines|exploration)\b", "15104020"),
    (r"\b(oil|gas|petroleum|royalty trust)\b", "10102020"),
    (r"\b(insurance|assurance|underwriters)\b", "40301040"),
    (r"\b(semiconductor|microelectronics)\b", "45301020"),
    (r"\b(software|cloud|\.ai\b|artificial intelligence|cyber)\b", "45103010"),
    (r"\b(medical|health ?care|surgical|diagnostics|dental|medtech)\b", "35101010"),
    (r"\b(restaurant|grill|pizza|burger)\b", "25301040"),
    (r"\b(foods?|beverage|brewing|winery|distill)", "30202030"),
    (r"\b(solar|wind|renewable|hydrogen|clean energy|battery|lithium)\b", "20104010"),
    (r"\b(airlines?|aviation|aerospace|defense)\b", "20101010"),
    (r"\b(shipping|tankers|maritime)\b", "20303010"),
    (r"\b(cannabis|hemp|cbd)\b", "35202010"),
    (r"\b(entertainment|studios|esports)\b", "50202010"),
    (r"\b(telecom|wireless)\b", "50101020"),
    (r"\b(utilities|electric (power|co)|power (co|corp)|water (co|corp|works))\b", "55103010"),
    (r"\b(chemicals?)\b", "15101050"),
    (r"\b(steel)\b", "15104050"),
    (r"\b(automotive|motors)\b", "25101010"),
    (r"\b(construction|engineering|infrastructure|builders?)\b", "20103010"),
    (r"\b(logistics|freight|trucking)\b", "20301010"),
    (r"\b(education|learning|academy|university)\b", "25302010"),
    (r"\b(hotels?|resorts?|hospitality|casino)\b", "25301020"),
    (r"\b(apparel|fashion)\b", "25203010"),
    (r"\b(industries|industrial|manufacturing|machinery)\b", "20106020"),
]

# Names that are not common equity at all, whatever Polygon's `type` says: notes, bonds,
# preferreds, depositary shares, ETFs, grantor and investment trusts, funds. GICS is not
# assigned to funds or ETFs, and a bond has no sector of its own.
NON_EQUITY_NAME = re.compile(
    r"(\d+(\.\d+)?%|\b(senior|subordinated|junior|fixed[- ]rate|floating[- ]rate|convertible|perpetual)\b)[^|]{0,60}\b(notes?|debentures?|bonds?)\b"
    r"|\b(notes?|debentures?|bonds?) due\b|\bfirst mortgage bonds?\b|\btrust preferred\b|\bpreferred (stock|shares?|units?)\b|\bpreference shares?\b"
    r"|\bdepositary (shares?|receipts?)\b|\bunits? of beneficial interest\b|\betfs?\b|\betns?\b|\bexchange[- ]traded (fund|note)\b"
    r"|\bclosed[- ]end fund\b|\bfund\b|\bfunds\b|\bunit investment trust\b|(?<!real estate )(?<!realty )\binvestment trust\b|\binvestment tr\b|\binv tr\b"
    r"|\b(bitcoin|ethereum|ether|solana|xrp|litecoin|crypto|gold|silver|platinum|palladium|crude oil|natural gas|commodity|treasury) (trust|fund|etf)\b"
    r"|\bwarrants?\b|\bsubscription rights?\b|\bcontingent value rights?\b"
    r"|\b(income|opportunit\w*|dividend|long/short|focus|total return|municipal|tax[- ]exempt|strateg\w*|allocation)\b(?![^|]{0,60}\b(realty|propert\w*|mortgage|royalty|reit|bancorp|bank))[^|]{0,60}\btrust\b",
    re.IGNORECASE,
)
SHELL_NAME = re.compile(r"\b(acquisition (corp|co|company|corporation|holdings)|spac\b|blank check|special purpose acquisition)", re.IGNORECASE)


@dataclass
class Evidence:
    asset_id: int
    ticker: str
    name: str | None
    description: str | None
    sic_code: str | None
    sic_description: str | None
    yahoo_industry: str | None  # Yahoo level-2 node name, e.g. "Banks - Regional"


@dataclass
class Placement:
    code: str | None            # GICS sub-industry code, or None when not placed
    confidence: float | None
    method: str                 # yahoo+sic | yahoo | sic | keywords | shell | none
    rationale: str
    excluded_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def _text(ev: Evidence) -> str:
    return " ".join(x for x in (ev.name, ev.sic_description, ev.description) if x).lower()


def _refine(default: str, refinements: list[tuple[str, str]], text: str) -> tuple[str, str | None]:
    for pattern, code in refinements:
        if re.search(pattern, text):
            return code, pattern
    return default, None


def sic_candidate(ev: Evidence, text: str) -> tuple[str | None, str]:
    """GICS code from the SIC code alone, with how it was derived."""
    sic = (ev.sic_code or "").strip()
    if not sic:
        return None, ""
    sic = sic.zfill(4)
    if sic in SIC_TO_GICS:
        default, refinements = SIC_TO_GICS[sic]
        if default == SHELL:
            return SHELL, f"sic {sic} blank check"
        code, pat = _refine(default, refinements, text)
        return code, f"sic {sic}->{code}" + (f" (refined: /{pat}/)" if pat else "")
    prefix = SIC_PREFIX_TO_GICS.get(sic[:2])
    if prefix:
        return prefix, f"sic {sic} (2-digit prefix)->{prefix}"
    return None, f"sic {sic} unmapped"


def yahoo_candidate(ev: Evidence, text: str) -> tuple[str | None, str]:
    name = (ev.yahoo_industry or "").strip()
    if not name:
        return None, ""
    if name not in YAHOO_TO_GICS:
        return None, f"yahoo '{name}' unmapped"
    default, refinements = YAHOO_TO_GICS[name]
    if default == SHELL:
        return SHELL, f"yahoo '{name}'"
    code, pat = _refine(default, refinements, text)
    return code, f"yahoo '{name}'->{code}" + (f" (refined: /{pat}/)" if pat else "")


def classify(ev: Evidence) -> Placement:
    """Place one security. Yahoo leads, SIC confirms or fills, keywords are last resort."""
    text = _text(ev)
    if ev.name and NON_EQUITY_NAME.search(ev.name):
        return Placement(None, None, "non_equity", f"name reads as a note/fund/preferred: {ev.name!r}", excluded_reason="not_common_equity")
    y_code, y_why = yahoo_candidate(ev, text)
    s_code, s_why = sic_candidate(ev, text)
    if not y_code and ev.name and SHELL_NAME.search(ev.name):
        return Placement(None, None, "shell", f"name /{SHELL_NAME.pattern}/", excluded_reason="shell_company")

    # Shell companies: the methodology assigns no GICS. Yahoo's word is current; a SIC
    # 6770 with a real Yahoo industry is a de-SPACed company and follows Yahoo.
    if y_code == SHELL:
        return Placement(None, None, "shell", y_why, excluded_reason="shell_company")
    if s_code == SHELL:
        if y_code:
            return Placement(y_code, 0.6, "yahoo", f"{y_why}; {s_why} ignored (de-SPAC)")
        if re.search(r"\b(acquisition|spac|blank check)\b", (ev.name or "").lower()) or not re.search(r"\b(revenue|customers|products|operates|manufactur|provides|sells)\b", text):
            return Placement(None, None, "shell", s_why, excluded_reason="shell_company")
        s_code, s_why = None, s_why + " (ignored: description reads as an operating business)"

    if y_code and s_code:
        if y_code == s_code:
            return Placement(y_code, 0.9, "yahoo+sic", f"{y_why}; {s_why}; agree")
        if y_code[:4] == s_code[:4]:
            return Placement(y_code, 0.75, "yahoo+sic", f"{y_why}; {s_why}; same industry group, Yahoo wins")
        if y_code[:2] == s_code[:2]:
            return Placement(y_code, 0.7, "yahoo+sic", f"{y_why}; {s_why}; same sector, Yahoo wins")
        return Placement(y_code, 0.6, "yahoo", f"{y_why}; {s_why}; sectors differ, Yahoo wins (current)")
    if y_code:
        return Placement(y_code, 0.65, "yahoo", y_why)
    if s_code:
        return Placement(s_code, 0.55, "sic", s_why)

    for pattern, code in NAME_RULES:
        if re.search(pattern, (ev.name or "").lower()):
            return Placement(code, 0.35, "keywords", f"name /{pattern}/->{code}")
    return Placement(None, None, "none", "no Yahoo industry, no SIC code, no usable words", excluded_reason="no_evidence")


def all_target_codes() -> set[str]:
    codes: set[str] = set()
    for default, refs in list(YAHOO_TO_GICS.values()) + list(SIC_TO_GICS.values()):
        if default != SHELL:
            codes.add(default)
        codes.update(c for _, c in refs)
    codes.update(SIC_PREFIX_TO_GICS.values())
    codes.update(c for _, c in NAME_RULES)
    return codes
