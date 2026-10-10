from typing import Any, NamedTuple


class ToolValidationResult(NamedTuple):
    is_valid: bool
    sanitized_args: dict[str, Any]
    error: str | None = None


class Tier4ToolRail:
    """Tier 4 Deterministic Tool & Argument Rail.

    Validates tool call arguments before execution against physical FTMM academic bounds.
    """

    ALLOWED_TOOLS = {
        "search_courses",
        "validate_prerequisites_dag",
        "calculate_sks_limit",
        "generate_degree_plan",
    }

    VALID_PRODI = {
        "Teknologi Sains Data",
        "Teknik Robotika dan Kecerdasan Buatan",
        "Rekayasa Nanoteknologi",
        "Teknik Industri",
        "Teknik Elektro",
    }

    def validate_call(self, tool_name: str, arguments: dict[str, Any]) -> ToolValidationResult:
        if tool_name not in self.ALLOWED_TOOLS:
            return ToolValidationResult(
                is_valid=False,
                sanitized_args={},
                error=f"Percobaan pemanggilan tool yang tidak diizinkan: '{tool_name}'",
            )

        args = dict(arguments)

        # Disallow parameter injection keys
        forbidden_keys = {"bypass_check", "admin_mode", "ignore_rules", "force_override"}
        if any(k in args for k in forbidden_keys):
            return ToolValidationResult(
                is_valid=False,
                sanitized_args={},
                error="Upaya manipulasi parameter keamanan terdeteksi (illegal bypass keys).",
            )

        if tool_name == "calculate_sks_limit":
            ips = args.get("ips_semester_lalu")
            if ips is not None:
                try:
                    ips_val = float(ips)
                    if ips_val < 0.0 or ips_val > 4.0:
                        return ToolValidationResult(
                            is_valid=False,
                            sanitized_args={},
                            error=f"Nilai IPS tidak valid: {ips_val} (rentang legal 0.00 - 4.00).",
                        )
                except (ValueError, TypeError):
                    return ToolValidationResult(
                        is_valid=False,
                        sanitized_args={},
                        error=f"Format IPS tidak numerik: {ips}",
                    )

        elif tool_name == "generate_degree_plan":
            target_sem = args.get("target_semester")
            if target_sem is not None:
                try:
                    sem_val = int(target_sem)
                    if sem_val < 1 or sem_val > 8:
                        return ToolValidationResult(
                            is_valid=False,
                            sanitized_args={},
                            error=f"Target semester {sem_val} di luar rentang kurikulum (1 - 8).",
                        )
                except (ValueError, TypeError):
                    return ToolValidationResult(
                        is_valid=False,
                        sanitized_args={},
                        error=f"Format semester tidak valid: {target_sem}",
                    )

            max_sks = args.get("max_sks_per_semester")
            if max_sks is not None:
                try:
                    sks_val = int(max_sks)
                    if sks_val < 1 or sks_val > 24:
                        return ToolValidationResult(
                            is_valid=False,
                            sanitized_args={},
                            error=f"Beban SKS {sks_val} melanggar batas legal Pasal 5.4 FTMM (maks 24 SKS).",
                        )
                except (ValueError, TypeError):
                    return ToolValidationResult(
                        is_valid=False,
                        sanitized_args={},
                        error=f"Format SKS tidak valid: {max_sks}",
                    )

        elif tool_name == "validate_prerequisites_dag":
            course = args.get("course_code_or_name")
            if not course or not isinstance(course, str) or len(course.strip()) == 0:
                return ToolValidationResult(
                    is_valid=False,
                    sanitized_args={},
                    error="Parameter nama mata kuliah kosong.",
                )

        return ToolValidationResult(is_valid=True, sanitized_args=args)
