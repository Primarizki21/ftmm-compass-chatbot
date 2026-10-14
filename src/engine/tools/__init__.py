from typing import Any
from src.engine.tools.academic_tools import (
    search_courses,
    validate_prerequisites_dag,
    calculate_sks_limit,
    generate_degree_plan,
    get_courses_by_semester,
    format_course_catalog_response,
    format_degree_plan_response,
    compress_degree_plan_for_llm,
)

# Tool Schemas for native Ollama function-calling (OpenAI tool format)
ACADEMIC_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": "Mencari silabus dan data mata kuliah FTMM berdasarkan topik/kata kunci spesifik (untuk daftar lengkap mata kuliah per semester, gunakan get_courses_by_semester).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Kata kunci pencarian, topik, atau nama mata kuliah."
                    },
                    "prodi": {
                        "type": "string",
                        "description": "Nama prodi FTMM untuk menyaring hasil (opsional)."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Jumlah rekomendasi mata kuliah yang diinginkan (default 3)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_prerequisites_dag",
            "description": "Memeriksa apakah mahasiswa berhak mengambil suatu mata kuliah berdasarkan riwayat mata kuliah yang sudah lulus (Prerequisite DAG validation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code_or_name": {
                        "type": "string",
                        "description": "Mata kuliah TARGET yang ingin diambil/diprogram mahasiswa (contoh: jika mahasiswa bertanya 'Bolehkah saya mengambil Pengolahan Citra Digital sebelum lulus Kalkulus I?', TARGET yang diperiksa adalah 'Pengolahan Citra Digital', BUKAN 'Kalkulus I'). Dilarang membalik target dengan prasyarat."
                    },
                    "passed_courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Daftar mata kuliah yang sudah lulus (opsional, default [])."
                    }
                },
                "required": ["course_code_or_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sks_limit",
            "description": "Menghitung batas beban studi SKS maksimum semester depan berdasarkan Indeks Prestasi Semester (IPS) sesuai Pasal 5.4 Panduan Akademik FTMM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ips": {
                        "type": "number",
                        "description": "Indeks Prestasi Semester (IPS) mahasiswa sebelumnya (rentang 0.00 hingga 4.00)."
                    }
                },
                "required": ["ips"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_degree_plan",
            "description": "Menghasilkan rancangan rencana studi (Degree Plan) dari semester saat ini sampai target lulus dengan validasi prasyarat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_studi": {
                        "type": "string",
                        "description": "Nama program studi resmi FTMM (misal: 'Teknologi Sains Data', 'Teknik Robotika dan Kecerdasan Buatan')."
                    },
                    "current_semester": {
                        "type": "integer",
                        "description": "Semester yang sedang ditempuh saat ini (1-8)."
                    },
                    "target_semester": {
                        "type": "integer",
                        "description": "Target semester kelulusan (7 atau 8)."
                    },
                    "passed_courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Daftar mata kuliah yang sudah lulus."
                    },
                    "focus_tracks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Daftar topik minat/spesialisasi yang diinginkan (misal: ['Machine Learning', 'Computer Vision'])."
                    },
                    "max_sks_per_sem": {
                        "type": "integer",
                        "description": "Batas maksimum SKS per semester (default 24)."
                    }
                },
                "required": ["current_semester"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_courses_by_semester",
            "description": "Mengambil daftar lengkap seluruh mata kuliah untuk program studi dan semester tertentu di FTMM secara deterministik (100% akurat tanpa batas top_k).",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_studi": {
                        "type": "string",
                        "description": "Program studi di FTMM (misal: 'TSD', 'TRKB', 'RN', 'TI', 'TE', 'Teknologi Sains Data', dll)."
                    },
                    "semester": {
                        "type": "integer",
                        "description": "Semester akademik yang dicari (1-8)."
                    },
                    "jenis": {
                        "type": "string",
                        "description": "Filter jenis mata kuliah opsional ('wajib', 'pilihan', atau kosongkan untuk semua)."
                    }
                },
                "required": ["program_studi", "semester"]
            }
        }
    }
]

TOOL_REGISTRY: dict[str, Any] = {
    "search_courses": search_courses,
    "validate_prerequisites_dag": validate_prerequisites_dag,
    "calculate_sks_limit": calculate_sks_limit,
    "generate_degree_plan": generate_degree_plan,
    "get_courses_by_semester": get_courses_by_semester,
}
