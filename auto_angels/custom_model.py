from datetime import datetime

class CustomModel:
    def __init__(self, model, dataset, numeric_attributes):
        self.model = model

        # Processar os atributos para incluir as datas ao invés da idade
        self.attributes_info = {}
        self.valid_options = {}  # Dicionário para armazenar opções válidas
        self.numeric_attributes = numeric_attributes  # Lista de atributos que precisam ser validados como numéricos

        for column in dataset.columns:
            if column == "age":
                self.attributes_info["mothers_birth_date"] = "Data"
                self.attributes_info["date_start_pregnancy"] = "Data"
                self.valid_options["mothers_birth_date"] = ["YYYY-MM-DD"]
                self.valid_options["date_start_pregnancy"] = ["YYYY-MM-DD"]
            elif column == "primeiro_pre_natal":
                self.attributes_info["date_first_prenatal"] = "Data"
                self.attributes_info["date_start_pregnancy"] = "Data"
                self.valid_options["date_first_prenatal"] = ["YYYY-MM-DD"]
                self.valid_options["date_start_pregnancy"] = ["YYYY-MM-DD"]
            elif column == "time_between_pregnancies":
                self.attributes_info["date_last_delivery"] = "Data"
                self.attributes_info["date_start_pregnancy"] = "Data"
                self.valid_options["date_last_delivery"] = ["YYYY-MM-DD"]
                self.valid_options["date_start_pregnancy"] = ["YYYY-MM-DD"]
            elif column == "target":
                continue
            else:
                self.attributes_info[column] = dataset[column].dtype
                self.valid_options[column] = dataset[column].unique().tolist()

        self.pre_processing_steps = [
            self.calculate_age,
            self.first_prenatal_weeks,
            self.calculate_time_between_pregnancies
        ]

    def calculate_age(self, birth_date, notification_date):
        birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
        notification_date = datetime.strptime(notification_date, "%Y-%m-%d")
        age = notification_date.year - birth_date.year - ((notification_date.month, notification_date.day) < (birth_date.month, birth_date.day))
        return age

    def first_prenatal_weeks(self, pregnancy_start_date, first_prenatal_date):
        pregnancy_start_date = datetime.fromisoformat(pregnancy_start_date)
        first_prenatal_date = datetime.fromisoformat(first_prenatal_date)

        weeks_count = abs((int)((pregnancy_start_date - first_prenatal_date).days / 7))

        return weeks_count

    def calculate_time_between_pregnancies(self, pregnancy_start_date, last_delivery_date):
        if last_delivery_date is None:
            return -1

        pregnancy_start_date = datetime.fromisoformat(pregnancy_start_date)
        last_delivery_date = datetime.fromisoformat(last_delivery_date)

        months_count = abs((int)((pregnancy_start_date - last_delivery_date).days / 30))

        if months_count > 12:
            months_count = 12

        return months_count

    def preprocess(self, data):
        processed_data = {}
        for attribute in self.attributes_info.keys():
            if attribute == "date_start_pregnancy":
                continue
            if attribute == "mothers_birth_date":
                processed_data["idade"] = self.calculate_age(data["mothers_birth_date"], data["date_start_pregnancy"])
            elif attribute == "date_first_prenatal":
                processed_data['primeiro_prenatal'] = self.first_prenatal_weeks(data["date_start_pregnancy"], data["date_first_prenatal"])
            elif attribute == "date_last_delivery":
                processed_data['time_between_pregnancy'] = self.calculate_time_between_pregnancies(data["date_start_pregnancy"], data["date_last_delivery"])
            else:
                value = data.get(attribute)
                if attribute in self.numeric_attributes:
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Invalid numeric value for {attribute}: {value}")
                else:
                    if value not in self.valid_options.get(attribute, [value]):
                        raise ValueError(f"Invalid value for {attribute}: {value}")
                processed_data[attribute] = value

        return processed_data