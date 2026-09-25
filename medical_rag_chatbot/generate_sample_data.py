import os
from pathlib import Path
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from config import DATA_RAW_PDFS, DATA_RAW_TABLES, DATA_RAW_IMAGES

def create_sample_pdfs():
    os.makedirs(DATA_RAW_PDFS, exist_ok=True)
    
    # 1. Policy Document: CLM-2024-001_policy.pdf
    doc1 = pymupdf.open()
    page1 = doc1.new_page()
    policy_text = """APEX HEALTHCARE INSURANCE - COMPREHENSIVE BENEFIT POLICY (2024)
Policy Number: POL-884920
Claim ID: CLM-2024-001
Effective Date: January 1, 2024

SECTION 1: COVERAGE OVERVIEW AND DEDUCTIBLES
1.1 Annual Deductible:
- Individual in-network deductible: $1,500 per calendar year.
- Family in-network deductible: $3,000 per calendar year.
- Out-of-network deductible: $3,500 individual / $7,000 family.

1.2 Out-of-Pocket Maximum:
- Maximum out-of-pocket for in-network medical care is capped at $5,000 for individuals and $10,000 for families.

SECTION 2: COPAYMENTS AND COINSURANCE
2.1 Primary Care Visits:
- Copay is $25 per office visit for participating network providers. Preventive health screenings are covered at 100% with no copay.

2.2 Specialist Consultations:
- In-network specialist copay is $40 per visit.
- Prior authorization from the designated primary care physician (PCP) is strictly required for outpatient specialist consultations (CPT code 99243 and 99213).
- Failure to obtain prior authorization prior to the consultation date will result in immediate claim denial under Plan B policy exclusions.

SECTION 3: INPATIENT AND EMERGENCY CARE
3.1 Emergency Room Services:
- Emergency room visit copay is $250, waived if admitted as an inpatient within 24 hours.
3.2 Inpatient Hospital Stay:
- Covered at 80% coinsurance after the annual deductible has been fully satisfied.
"""
    page1.insert_text((50, 60), policy_text, fontsize=11, fontname="helv")
    doc1.save(str(DATA_RAW_PDFS / "CLM-2024-001_policy.pdf"))
    doc1.close()

    # 2. Claim Denial & EOB: CLM-2024-002_eob.pdf
    doc2 = pymupdf.open()
    page2 = doc2.new_page()
    eob_text = """EXPLANATION OF BENEFITS (EOB) - NOTICE OF ADJUDICATION
Insurance Carrier: Apex Healthcare Solutions
Claim ID: CLM-2024-002
Patient ID: PT-90412
Date of Service: February 14, 2024
Provider: Metro Health Orthopedic Center

CLAIM SUMMARY AND DISPOSITION:
Claim Status: DENIED
Billed Amount: $1,450.00
Allowed Amount: $0.00
Patient Responsibility: $1,450.00

DENIAL REASON CODE & DETAILS:
Denial Code: EX-99213 / D-04
Denial Reason: Non-covered elective physical therapy modality.
Details: Physical therapy treatment code 97110 exceeded the allowable limit of 20 visits per calendar year without medical necessity recertification from an attending physiatrist.

APPEAL RIGHTS:
The member may file a written first-level appeal within 180 days of this notice. Supporting clinical notes, diagnostic imaging reports, and a physician letter of medical necessity must accompany the appeal packet.
"""
    page2.insert_text((50, 60), eob_text, fontsize=11, fontname="helv")
    doc2.save(str(DATA_RAW_PDFS / "CLM-2024-002_eob.pdf"))
    doc2.close()

    # 3. Inpatient Claim Form: CLM-2024-003_claim_form.pdf
    doc3 = pymupdf.open()
    page3 = doc3.new_page()
    claim_text = """HEALTH INSURANCE CLAIM FORM (HCFA-1500)
Claim ID: CLM-2024-003
Patient ID: PT-78219
Insured Group Number: GRP-55201
Date of Service: March 10, 2024
Facility: Saint Jude Community Hospital

DIAGNOSIS & CLINICAL PROCEDURE CODES:
Primary ICD-10 Code: I10 (Essential primary hypertension)
Secondary ICD-10 Code: E11.9 (Type 2 diabetes mellitus without complications)
Procedure CPT Code: 99222 (Initial hospital inpatient care, moderate complexity)
Charge Billed: $3,200.00

PRIOR AUTHORIZATION STATUS:
Prior Auth Number: PA-99281-AUTH
Status: Approved on March 08, 2024 by Medical Review Board
Approved Length of Stay: 3 Days
Attending Physician: Dr. Marcus Vance, MD (NPI: 1982736451)
"""
    page3.insert_text((50, 60), claim_text, fontsize=11, fontname="helv")
    doc3.save(str(DATA_RAW_PDFS / "CLM-2024-003_claim_form.pdf"))
    doc3.close()
    print("Created sample PDF documents.")

def create_sample_tables():
    os.makedirs(DATA_RAW_TABLES, exist_ok=True)
    
    # 1. Billing CSV: CLM-2024-001_billing.csv
    billing_data = [
        {"claim_id": "CLM-2024-001", "service_date": "2024-01-15", "cpt_code": "99213", "description": "Office outpatient visit 20-29 min", "charge_amount": 185.00, "allowed_amount": 0.00, "status": "DENIED", "denial_code": "NO_PREAUTH"},
        {"claim_id": "CLM-2024-001", "service_date": "2024-01-15", "cpt_code": "93000", "description": "Electrocardiogram routine ECG 12 leads", "charge_amount": 95.00, "allowed_amount": 75.00, "status": "PAID", "denial_code": "NONE"},
        {"claim_id": "CLM-2024-001", "service_date": "2024-01-15", "cpt_code": "80053", "description": "Comprehensive metabolic panel blood test", "charge_amount": 120.00, "allowed_amount": 90.00, "status": "PAID", "denial_code": "NONE"},
        {"claim_id": "CLM-2024-002", "service_date": "2024-02-14", "cpt_code": "97110", "description": "Therapeutic exercises physical therapy", "charge_amount": 250.00, "allowed_amount": 0.00, "status": "DENIED", "denial_code": "MAX_VISITS_EXCEEDED"},
        {"claim_id": "CLM-2024-002", "service_date": "2024-02-14", "cpt_code": "97010", "description": "Application of hot cold packs therapy", "charge_amount": 45.00, "allowed_amount": 0.00, "status": "DENIED", "denial_code": "NON_COVERED_MODALITY"},
        {"claim_id": "CLM-2024-003", "service_date": "2024-03-10", "cpt_code": "99222", "description": "Initial hospital inpatient care moderate", "charge_amount": 850.00, "allowed_amount": 680.00, "status": "PAID", "denial_code": "NONE"},
        {"claim_id": "CLM-2024-003", "service_date": "2024-03-11", "cpt_code": "36415", "description": "Routine venipuncture blood collection", "charge_amount": 35.00, "allowed_amount": 25.00, "status": "PAID", "denial_code": "NONE"},
        {"claim_id": "CLM-2024-004", "service_date": "2024-04-05", "cpt_code": "71045", "description": "Chest X-ray single view frontal", "charge_amount": 160.00, "allowed_amount": 120.00, "status": "PAID", "denial_code": "NONE"},
        {"claim_id": "CLM-2024-004", "service_date": "2024-04-05", "cpt_code": "94010", "description": "Spirometry pulmonary function test", "charge_amount": 210.00, "allowed_amount": 175.00, "status": "PAID", "denial_code": "NONE"},
        {"claim_id": "CLM-2024-005", "service_date": "2024-05-18", "cpt_code": "99284", "description": "Emergency department visit high severity", "charge_amount": 1450.00, "allowed_amount": 1160.00, "status": "PAID", "denial_code": "NONE"}
    ]
    df = pd.DataFrame(billing_data)
    df.to_csv(DATA_RAW_TABLES / "CLM-2024-001_billing.csv", index=False)
    print("Created sample billing table CSV.")

def create_sample_images():
    os.makedirs(DATA_RAW_IMAGES, exist_ok=True)
    
    # 1. Prescription Scan Image: CLM-2024-001_prescription.jpg
    img1 = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    # Header border
    draw1.rectangle([(20, 20), (780, 580)], outline=(30, 64, 175), width=3)
    draw1.text((50, 40), "MEDICAL CENTER PHARMACY - RX ORDER", fill=(30, 64, 175))
    draw1.text((50, 80), "Claim ID: CLM-2024-001 | Date: 2024-01-16", fill=(0, 0, 0))
    draw1.text((50, 120), "Patient Name: <PERSON> | Rx Number: RX-774021", fill=(0, 0, 0))
    draw1.text((50, 170), "Prescription: Lisinopril 20mg Oral Tablet", fill=(0, 0, 0))
    draw1.text((50, 210), "Directions: Take 1 tablet daily every morning for hypertension control", fill=(0, 0, 0))
    draw1.text((50, 250), "Quantity: 90 Tablets (90-day supply) | Refills: 3 Refills Authorized", fill=(0, 0, 0))
    draw1.text((50, 290), "NDC Code: 68180-514-01 | DAW: 0 (Generic Substitution Permitted)", fill=(0, 0, 0))
    draw1.text((50, 340), "Copay Collected: $10.00 Tier 1 Preferred Generic", fill=(0, 0, 0))
    draw1.text((50, 400), "Prescribing Physician: Dr. Elena Rostova, MD (DEA: BR9876543)", fill=(0, 0, 0))
    img1.save(DATA_RAW_IMAGES / "CLM-2024-001_prescription.jpg")

    # 2. Medical Scan Report Image: CLM-2024-002_scan.png
    img2 = Image.new("RGB", (800, 600), color=(248, 250, 252))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([(20, 20), (780, 580)], outline=(15, 118, 110), width=3)
    draw2.text((50, 40), "RADIOLOGY & DIAGNOSTIC IMAGING REPORT", fill=(15, 118, 110))
    draw2.text((50, 80), "Claim ID: CLM-2024-002 | Study Date: 2024-02-12", fill=(0, 0, 0))
    draw2.text((50, 120), "Exam Type: MRI Right Knee Without Contrast (CPT 73721)", fill=(0, 0, 0))
    draw2.text((50, 170), "Clinical History: Chronic right knee pain following sports injury", fill=(0, 0, 0))
    draw2.text((50, 210), "Findings: Mild joint effusion. No acute ligamentous tear in ACL or PCL.", fill=(0, 0, 0))
    draw2.text((50, 250), "Impression: Moderate medial meniscus degenerative tear at posterior horn.", fill=(0, 0, 0))
    draw2.text((50, 300), "Recommendation: Conservative management and structured physical therapy.", fill=(0, 0, 0))
    draw2.text((50, 350), "Prior Auth Match: PA-MRI-66321 Approved for Right Knee", fill=(0, 0, 0))
    draw2.text((50, 400), "Radiologist: Dr. Arthur Pendelton, MD (Board Certified)", fill=(0, 0, 0))
    img2.save(DATA_RAW_IMAGES / "CLM-2024-002_scan.png")

    # 3. Prescription Image: CLM-2024-003_prescription.jpg
    img3 = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([(20, 20), (780, 580)], outline=(194, 65, 12), width=3)
    draw3.text((50, 40), "OUTPATIENT PHARMACY PRESCRIPTION ORDER", fill=(194, 65, 12))
    draw3.text((50, 80), "Claim ID: CLM-2024-003 | Date: 2024-03-12", fill=(0, 0, 0))
    draw3.text((50, 120), "Medication: Metformin Hydrochloride 500mg Extended Release", fill=(0, 0, 0))
    draw3.text((50, 170), "Instructions: Take 1 tablet by mouth twice daily with meals", fill=(0, 0, 0))
    draw3.text((50, 220), "Quantity: 60 Tablets | Refills: 5 Refills", fill=(0, 0, 0))
    draw3.text((50, 270), "Therapeutic Class: Oral Hypoglycemic Agent", fill=(0, 0, 0))
    draw3.text((50, 320), "Insurance Coverage: Tier 1 Formulary Approved Copay $5.00", fill=(0, 0, 0))
    draw3.text((50, 380), "Dispensing Pharmacist: Sarah Jenkins, PharmD", fill=(0, 0, 0))
    img3.save(DATA_RAW_IMAGES / "CLM-2024-003_prescription.jpg")
    print("Created sample image scan documents.")

if __name__ == "__main__":
    create_sample_pdfs()
    create_sample_tables()
    create_sample_images()
    print("All sample multimodal documents created successfully!")
