def validate_input(data):
    cep = "".join(char for char in str(data.get("cep", "")) if char.isdigit())
    if not cep:
        raise ValueError("CEP invalido")

    if float(data["area_do_terreno_m2"]) <= 0:
        raise ValueError("Area invalida")

    if "ano" in data and not 1900 <= int(data["ano"]) <= 2999:
        raise ValueError("Ano invalido")

    if "mes" in data and not 1 <= int(data["mes"]) <= 12:
        raise ValueError("Mes invalido")


def validate_output(value):
    if value < 0:
        raise ValueError("Saida invalida do modelo")

    return value
