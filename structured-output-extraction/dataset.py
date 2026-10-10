"""Dataset generation module for contact information extraction evaluation.

Generates a deterministic, reproducible evaluation benchmark of labeled samples
saved as data/evaluation.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Default path to evaluation dataset
DATASET_PATH = Path(__file__).resolve().parent / "data" / "evaluation.jsonl"


def get_raw_evaluation_samples() -> List[Dict[str, Any]]:
    """Return comprehensive list of labeled test samples.

    Categories:
    - all_fields: All four fields (name, email, phone, job_title) are explicitly present.
    - partial_fields: 1, 2, or 3 fields are present; missing fields are None.
    - no_target_fields: Zero contact fields present (weather, error logs, recipes, announcements).
    - ambiguous: Distractors, order numbers, versions, handles, false emails, etc.
    """
    samples: List[Dict[str, Any]] = [
        # =========================================================================
        # Category: all_fields (16 samples)
        # =========================================================================
        {
            "sample_id": "SAMPLE-001",
            "category": "all_fields",
            "input_text": "Hi, I am Priya Sharma. You can reach me at priya@example.com or call +91 98765 43210. I work as a data analyst.",
            "expected_output": {
                "name": "Priya Sharma",
                "email": "priya@example.com",
                "phone": "+91 98765 43210",
                "job_title": "data analyst",
            },
        },
        {
            "sample_id": "SAMPLE-002",
            "category": "all_fields",
            "input_text": "Alexander Wright | Senior DevOps Engineer\nEmail: a.wright@cloudnet.org\nDirect: +1 (555) 321-9876\nSan Francisco, CA",
            "expected_output": {
                "name": "Alexander Wright",
                "email": "a.wright@cloudnet.org",
                "phone": "+1 (555) 321-9876",
                "job_title": "Senior DevOps Engineer",
            },
        },
        {
            "sample_id": "SAMPLE-003",
            "category": "all_fields",
            "input_text": "Please forward the project proposal to our Chief Technology Officer, Marcus Vance (marcus.vance@techcorp.io, tel: 415-555-0143).",
            "expected_output": {
                "name": "Marcus Vance",
                "email": "marcus.vance@techcorp.io",
                "phone": "415-555-0143",
                "job_title": "Chief Technology Officer",
            },
        },
        {
            "sample_id": "SAMPLE-004",
            "category": "all_fields",
            "input_text": "Warm regards,\nElena Rostova\nProduct Designer\nMobile: +44 20 7946 0991\nelena.rostova@designstudio.co.uk",
            "expected_output": {
                "name": "Elena Rostova",
                "email": "elena.rostova@designstudio.co.uk",
                "phone": "+44 20 7946 0991",
                "job_title": "Product Designer",
            },
        },
        {
            "sample_id": "SAMPLE-005",
            "category": "all_fields",
            "input_text": "Candidate: Sarah Jenkins\nApplied Position: Machine Learning Researcher\nContact Info: s_jenkins@ai-labs.org | 617-555-0199",
            "expected_output": {
                "name": "Sarah Jenkins",
                "email": "s_jenkins@ai-labs.org",
                "phone": "617-555-0199",
                "job_title": "Machine Learning Researcher",
            },
        },
        {
            "sample_id": "SAMPLE-006",
            "category": "all_fields",
            "input_text": "Contact our Lead Architect Hiroshi Tanaka via hiroshi.tanaka@tokyosoft.jp or telephone 03-5555-0182.",
            "expected_output": {
                "name": "Hiroshi Tanaka",
                "email": "hiroshi.tanaka@tokyosoft.jp",
                "phone": "03-5555-0182",
                "job_title": "Lead Architect",
            },
        },
        {
            "sample_id": "SAMPLE-007",
            "category": "all_fields",
            "input_text": "Dr. Beatrice Dupont, Clinical Pharmacologist. Correspondence: bdupont@healthcare.fr, cell 06-12-34-56-78.",
            "expected_output": {
                "name": "Dr. Beatrice Dupont",
                "email": "bdupont@healthcare.fr",
                "phone": "06-12-34-56-78",
                "job_title": "Clinical Pharmacologist",
            },
        },
        {
            "sample_id": "SAMPLE-008",
            "category": "all_fields",
            "input_text": "Name: Carlos Mendoza\nRole: VP of Sales\nPhone: +52 55 1234 5678\nMail: carlos.mendoza@latamvent.com",
            "expected_output": {
                "name": "Carlos Mendoza",
                "email": "carlos.mendoza@latamvent.com",
                "phone": "+52 55 1234 5678",
                "job_title": "VP of Sales",
            },
        },
        {
            "sample_id": "SAMPLE-009",
            "category": "all_fields",
            "input_text": "Billing inquiry contact: Maya Lin, Financial Analyst. Call +1-800-555-4321 or write to lin_maya@fintechcorp.net.",
            "expected_output": {
                "name": "Maya Lin",
                "email": "lin_maya@fintechcorp.net",
                "phone": "+1-800-555-4321",
                "job_title": "Financial Analyst",
            },
        },
        {
            "sample_id": "SAMPLE-010",
            "category": "all_fields",
            "input_text": "I am Robert Thorne, working as an Information Security Specialist. Ring me at 020 7946 0123 or reach out at rthorne@secops.co.uk.",
            "expected_output": {
                "name": "Robert Thorne",
                "email": "rthorne@secops.co.uk",
                "phone": "020 7946 0123",
                "job_title": "Information Security Specialist",
            },
        },
        {
            "sample_id": "SAMPLE-011",
            "category": "all_fields",
            "input_text": "Speaker profile: Fatima Al-Mansoor, Quantum Computing Scientist. Direct inquiries: fatima@q-systems.ae / +971 4 321 0000.",
            "expected_output": {
                "name": "Fatima Al-Mansoor",
                "email": "fatima@q-systems.ae",
                "phone": "+971 4 321 0000",
                "job_title": "Quantum Computing Scientist",
            },
        },
        {
            "sample_id": "SAMPLE-012",
            "category": "all_fields",
            "input_text": "Resume Summary: David O'Connor. Full Stack Developer. Phone: 312-555-7890. Email: david.oconnor@devmail.org.",
            "expected_output": {
                "name": "David O'Connor",
                "email": "david.oconnor@devmail.org",
                "phone": "312-555-7890",
                "job_title": "Full Stack Developer",
            },
        },
        {
            "sample_id": "SAMPLE-013",
            "category": "all_fields",
            "input_text": "For interviews contact our Talent Acquisition Lead, Zoe Kravitz at zoe.k@recruitmenthq.com or +1 408 555 9012.",
            "expected_output": {
                "name": "Zoe Kravitz",
                "email": "zoe.k@recruitmenthq.com",
                "phone": "+1 408 555 9012",
                "job_title": "Talent Acquisition Lead",
            },
        },
        {
            "sample_id": "SAMPLE-014",
            "category": "all_fields",
            "input_text": "From: Liam Murphy <liam.murphy@dublintech.ie>\nRole: Database Administrator\nContact: +353 1 496 0123",
            "expected_output": {
                "name": "Liam Murphy",
                "email": "liam.murphy@dublintech.ie",
                "phone": "+353 1 496 0123",
                "job_title": "Database Administrator",
            },
        },
        {
            "sample_id": "SAMPLE-015",
            "category": "all_fields",
            "input_text": "Chief Executive Officer: Olivia Chen | ochen@acmeglobal.com | Desk: 212-555-6677",
            "expected_output": {
                "name": "Olivia Chen",
                "email": "ochen@acmeglobal.com",
                "phone": "212-555-6677",
                "job_title": "Chief Executive Officer",
            },
        },
        {
            "sample_id": "SAMPLE-016",
            "category": "all_fields",
            "input_text": "Hello, this is Arthur Dent, Space Systems Consultant. Email is adent@galactic.org and voice comm is 555-424-2424.",
            "expected_output": {
                "name": "Arthur Dent",
                "email": "adent@galactic.org",
                "phone": "555-424-2424",
                "job_title": "Space Systems Consultant",
            },
        },

        # =========================================================================
        # Category: partial_fields (24 samples)
        # =========================================================================
        {
            "sample_id": "SAMPLE-017",
            "category": "partial_fields",
            "input_text": "Hey team, my name is John Smith and you can email me at john.smith@company.com.",
            "expected_output": {
                "name": "John Smith",
                "email": "john.smith@company.com",
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-018",
            "category": "partial_fields",
            "input_text": "Call Jessica at 555-987-6543 if you have questions regarding the shipment.",
            "expected_output": {
                "name": "Jessica",
                "email": None,
                "phone": "555-987-6543",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-019",
            "category": "partial_fields",
            "input_text": "Please welcome Michael Chang, who has joined us as our new Marketing Director.",
            "expected_output": {
                "name": "Michael Chang",
                "email": None,
                "phone": None,
                "job_title": "Marketing Director",
            },
        },
        {
            "sample_id": "SAMPLE-020",
            "category": "partial_fields",
            "input_text": "Direct inquiries can be sent to info@globalcorp.org.",
            "expected_output": {
                "name": None,
                "email": "info@globalcorp.org",
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-021",
            "category": "partial_fields",
            "input_text": "Customer hotline is 1-800-234-5678 available 24/7.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": "1-800-234-5678",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-022",
            "category": "partial_fields",
            "input_text": "The position of Senior Backend Engineer requires at least five years of experience.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": "Senior Backend Engineer",
            },
        },
        {
            "sample_id": "SAMPLE-023",
            "category": "partial_fields",
            "input_text": "Alice Walker, QA Automation Engineer. Contact: awalker@testinghub.io.",
            "expected_output": {
                "name": "Alice Walker",
                "email": "awalker@testinghub.io",
                "phone": None,
                "job_title": "QA Automation Engineer",
            },
        },
        {
            "sample_id": "SAMPLE-024",
            "category": "partial_fields",
            "input_text": "Contact Bob Davis at 206-555-0177 for technical support regarding the cluster.",
            "expected_output": {
                "name": "Bob Davis",
                "email": None,
                "phone": "206-555-0177",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-025",
            "category": "partial_fields",
            "input_text": "Our Operations Manager can be reached at ops-manager@logistics.com or 555-432-1098.",
            "expected_output": {
                "name": None,
                "email": "ops-manager@logistics.com",
                "phone": "555-432-1098",
                "job_title": "Operations Manager",
            },
        },
        {
            "sample_id": "SAMPLE-026",
            "category": "partial_fields",
            "input_text": "Thanks, Nathan Cole. Mobile: +1 (718) 555-0112.",
            "expected_output": {
                "name": "Nathan Cole",
                "email": None,
                "phone": "+1 (718) 555-0112",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-027",
            "category": "partial_fields",
            "input_text": "Send bug reports to security-officer@kernel.org. Phone: 408-555-2233.",
            "expected_output": {
                "name": None,
                "email": "security-officer@kernel.org",
                "phone": "408-555-2233",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-028",
            "category": "partial_fields",
            "input_text": "I am Samuel Green. I am currently working as a Solutions Architect.",
            "expected_output": {
                "name": "Samuel Green",
                "email": None,
                "phone": None,
                "job_title": "Solutions Architect",
            },
        },
        {
            "sample_id": "SAMPLE-029",
            "category": "partial_fields",
            "input_text": "Hannah Abbott, UX Researcher. Please schedule a chat via hannah.abbott@uxlab.org.",
            "expected_output": {
                "name": "Hannah Abbott",
                "email": "hannah.abbott@uxlab.org",
                "phone": None,
                "job_title": "UX Researcher",
            },
        },
        {
            "sample_id": "SAMPLE-030",
            "category": "partial_fields",
            "input_text": "Vikram Patel, VP of Engineering. Reach me at +91-22-6789-0123.",
            "expected_output": {
                "name": "Vikram Patel",
                "email": None,
                "phone": "+91-22-6789-0123",
                "job_title": "VP of Engineering",
            },
        },
        {
            "sample_id": "SAMPLE-031",
            "category": "partial_fields",
            "input_text": "Drop a line to chloe.bennet@marvelstudios.com anytime.",
            "expected_output": {
                "name": None,
                "email": "chloe.bennet@marvelstudios.com",
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-032",
            "category": "partial_fields",
            "input_text": "Call our hotline at +44 1632 960888.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": "+44 1632 960888",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-033",
            "category": "partial_fields",
            "input_text": "Regards,\nClara Oswald\nTeacher",
            "expected_output": {
                "name": "Clara Oswald",
                "email": None,
                "phone": None,
                "job_title": "Teacher",
            },
        },
        {
            "sample_id": "SAMPLE-034",
            "category": "partial_fields",
            "input_text": "Daniel Craig | d.craig@mi6.gov.uk | Secret Agent",
            "expected_output": {
                "name": "Daniel Craig",
                "email": "d.craig@mi6.gov.uk",
                "phone": None,
                "job_title": "Secret Agent",
            },
        },
        {
            "sample_id": "SAMPLE-035",
            "category": "partial_fields",
            "input_text": "My phone is 919-555-3344 and my email is emily.blunt@cinema.com.",
            "expected_output": {
                "name": None,
                "email": "emily.blunt@cinema.com",
                "phone": "919-555-3344",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-036",
            "category": "partial_fields",
            "input_text": "Dr. Strange, Surgeon. Cell: 212-555-9999.",
            "expected_output": {
                "name": "Dr. Strange",
                "email": None,
                "phone": "212-555-9999",
                "job_title": "Surgeon",
            },
        },
        {
            "sample_id": "SAMPLE-037",
            "category": "partial_fields",
            "input_text": "Signed by George Washington.",
            "expected_output": {
                "name": "George Washington",
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-038",
            "category": "partial_fields",
            "input_text": "Contact: support@developerportal.dev",
            "expected_output": {
                "name": None,
                "email": "support@developerportal.dev",
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-039",
            "category": "partial_fields",
            "input_text": "Grace Hopper, Computer Scientist.",
            "expected_output": {
                "name": "Grace Hopper",
                "email": None,
                "phone": None,
                "job_title": "Computer Scientist",
            },
        },
        {
            "sample_id": "SAMPLE-040",
            "category": "partial_fields",
            "input_text": "Lucas Scott (555-012-3456) left a message regarding the basketball schedule.",
            "expected_output": {
                "name": "Lucas Scott",
                "email": None,
                "phone": "555-012-3456",
                "job_title": None,
            },
        },

        # =========================================================================
        # Category: no_target_fields (15 samples)
        # =========================================================================
        {
            "sample_id": "SAMPLE-041",
            "category": "no_target_fields",
            "input_text": "The weather is pleasant today. I went for a walk in the park.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-042",
            "category": "no_target_fields",
            "input_text": "To bake sourdough bread, mix flour, water, and active starter. Allow the dough to ferment for 12 hours.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-043",
            "category": "no_target_fields",
            "input_text": "Error 503: Service Temporarily Unavailable. Please check back later.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-044",
            "category": "no_target_fields",
            "input_text": "The quarterly earnings meeting has been rescheduled to Thursday at 3:00 PM in Conference Room B.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-045",
            "category": "no_target_fields",
            "input_text": "Kubernetes cluster upgraded to version 1.30. Nodes are healthy and pods are running.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-046",
            "category": "no_target_fields",
            "input_text": "The quick brown fox jumps over the lazy dog.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-047",
            "category": "no_target_fields",
            "input_text": "Flight AA1024 from Dallas to Seattle is boarding at gate C14.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-048",
            "category": "no_target_fields",
            "input_text": "Reminder: Bring your laptop charger and notebook to tomorrow's workshop.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-049",
            "category": "no_target_fields",
            "input_text": "Stock index rose 1.4% following the central bank's announcement on interest rates.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-050",
            "category": "no_target_fields",
            "input_text": "Chapter 4: The principles of thermodynamic equilibrium and entropy.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-051",
            "category": "no_target_fields",
            "input_text": "SELECT id, created_at, status FROM orders WHERE total > 100.00 ORDER BY created_at DESC;",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-052",
            "category": "no_target_fields",
            "input_text": "The museum will be closed for maintenance between October 15 and October 20.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-053",
            "category": "no_target_fields",
            "input_text": "Apples, bananas, milk, eggs, whole grain bread, and olive oil.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-054",
            "category": "no_target_fields",
            "input_text": "No parking between 8:00 AM and 6:00 PM on street cleaning days.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-055",
            "category": "no_target_fields",
            "input_text": "Sunset is scheduled for 7:42 PM with zero chance of precipitation.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },

        # =========================================================================
        # Category: ambiguous (15 samples)
        # =========================================================================
        {
            "sample_id": "SAMPLE-056",
            "category": "ambiguous",
            "input_text": "Your order #98214-4321 has shipped via FedEx tracking 1234-5678-9012.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-057",
            "category": "ambiguous",
            "input_text": "Checkout our release notes at v2.4.1@stable-build for bug fixes.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-058",
            "category": "ambiguous",
            "input_text": "Follow our CEO on Twitter @tech_guru for daily startup insights.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-059",
            "category": "ambiguous",
            "input_text": "Invoice 2026-0045 was billed to Acme Corp for $4,500.00.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-060",
            "category": "ambiguous",
            "input_text": "Connect with me at john [at] domain [dot] com or call five-five-five zero-one-two-three.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-061",
            "category": "ambiguous",
            "input_text": "The city of San Francisco established 911 for emergency calls and 311 for municipal inquiries.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-062",
            "category": "ambiguous",
            "input_text": "Package dimensions: 10x12x8 inches, gross weight 4.2 kg, item code: SKU-990-2211.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-063",
            "category": "ambiguous",
            "input_text": "Alice in Wonderland is a novel by Lewis Carroll published in 1865.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-064",
            "category": "ambiguous",
            "input_text": "Repository migrated to https://github.com/organization/core-service.git",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-065",
            "category": "ambiguous",
            "input_text": "Security Alert: SSH login attempt from 192.168.1.105 port 22 blocked by firewall.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-066",
            "category": "ambiguous",
            "input_text": "Meeting room 'Newton' booked from 10:00-11:30 by Engineering Team.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-067",
            "category": "ambiguous",
            "input_text": "Please contact our support desk (support@acme.com) regarding ticket INC-481923.",
            "expected_output": {
                "name": None,
                "email": "support@acme.com",
                "phone": None,
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-068",
            "category": "ambiguous",
            "input_text": "Call 555-0199 for reservations. Open Tuesday through Sunday.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": "555-0199",
                "job_title": None,
            },
        },
        {
            "sample_id": "SAMPLE-069",
            "category": "ambiguous",
            "input_text": "Job opening: Staff Systems Architect. Location: Remote. Salary: Competitive.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": "Staff Systems Architect",
            },
        },
        {
            "sample_id": "SAMPLE-070",
            "category": "ambiguous",
            "input_text": "Review submitted by customer 'User_9821' giving product rating 5/5 stars.",
            "expected_output": {
                "name": None,
                "email": None,
                "phone": None,
                "job_title": None,
            },
        },
    ]
    return samples


def save_dataset(filepath: Path = DATASET_PATH) -> int:
    """Save the evaluation samples to a JSONL file.

    Idempotent and safe to run multiple times without duplicating records.

    Returns:
        Number of samples written.
    """
    samples = get_raw_evaluation_samples()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            line = json.dumps(sample, ensure_ascii=False)
            f.write(line + "\n")

    logger.info("Saved %d evaluation samples to %s", len(samples), filepath)
    return len(samples)


def load_dataset(filepath: Path = DATASET_PATH) -> List[Dict[str, Any]]:
    """Load evaluation samples from JSONL file.

    If file does not exist, automatically generates and saves it.
    """
    if not filepath.exists():
        logger.warning("Dataset file %s does not exist. Generating now...", filepath)
        save_dataset(filepath)

    samples: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            sample = json.loads(line_str)
            samples.append(sample)

    return samples


def main() -> None:
    """CLI entrypoint to generate and display dataset statistics."""
    count = save_dataset(DATASET_PATH)
    samples = load_dataset(DATASET_PATH)

    categories: Dict[str, int] = {}
    for s in samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("=" * 60)
    print("STRUCTURED EXTRACTION EVALUATION DATASET")
    print("=" * 60)
    print(f"Output File:     {DATASET_PATH}")
    print(f"Total Samples:   {count}")
    print("Breakdown by Category:")
    for cat, cat_count in sorted(categories.items()):
        print(f"  - {cat:<20}: {cat_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
