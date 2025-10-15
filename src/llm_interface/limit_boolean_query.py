import re

def limit_boolean_query(query, max_parens=3, max_or=5, max_quotes=2):
    """
    검색식 제한 적용 (DB 안전용)
    - 괄호 최대 max_parens
    - 괄호 내부 OR 최대 max_or
    - 따옴표("") 최대 max_quotes
    """
    quotes = re.findall(r'"[^"]*"', query)
    if len(quotes) > max_quotes:
        for q in quotes[max_quotes:]:
            query = query.replace(q, q.strip('"'))

    paren_groups = re.findall(r'\([^\(\)]*\)', query)
    keep_groups = paren_groups[:max_parens]
    overflow_groups = paren_groups[max_parens:]

    for pg in keep_groups:
        query = query.replace(pg, f"__KEEP__{pg}__KEEP__", 1)

    for pg in overflow_groups:
        inner = pg.strip('()')
        query = query.replace(pg, inner, 1)

    def limit_or_inside_pg(pg):
        parts = pg.strip('()').split('|')
        if len(parts) > max_or:
            return '(' + '|'.join(parts[:max_or]) + ')'
        return pg

    query = re.sub(r'\([^\(\)]*\)', lambda m: limit_or_inside_pg(m.group(0)), query)

    keep_parts = re.findall(r'__KEEP__\([^\(\)]*\)__KEEP__', query)
    for i, part in enumerate(keep_parts):
        keep_parts[i] = part.replace('__KEEP__', '')
    if keep_parts:
        and_part = ' AND '.join(keep_parts)
        query = re.sub(r'__KEEP__\([^\(\)]*\)__KEEP__', '', query)
        query = f"{and_part} {query}"

    # 5️⃣ 괄호 제거 후 남은 OR 연결 구조 보정
    query = re.sub(r'\s+', ' ', query.strip())
    query = query.replace('AND |', 'AND')  # 잘못된 구문 정리
    query = query.replace('||', '|')

    return query.strip()
