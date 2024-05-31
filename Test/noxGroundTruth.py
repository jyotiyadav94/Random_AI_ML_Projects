from datasets.dataset_noxGroundTruth import data

class NOxDataAnalyzer:
    def __init__(self, data):
        self.data = data

    def num_stations(self):
        return len(self.data)

    def nox_reading_5th_station(self):
        return self.data[4][1][0]

    def nox_reading_last_station(self):
        return self.data[-1][1][0]

    def sum_nox_w_stations(self):
        return sum(int(station[1][0]) for station in self.data if station[0][0].startswith('W'))

    def average_nox_reading(self):
        num_stations = self.num_stations()
        total_nox = sum(int(station[1][0]) * (1 + int(station[2][0]) / 100) for station in self.data)
        return total_nox / num_stations

# Create an instance of the NOxDataAnalyzer class
nox_analyzer = NOxDataAnalyzer(data)

#printing results
print("Total number of stations:", nox_analyzer.num_stations())
print("NOx reading from the 5th station:", nox_analyzer.nox_reading_5th_station())
print("NOx reading from the last station:", nox_analyzer.nox_reading_last_station())
print("Total sum of NOx on sites with station names starting with 'W':", nox_analyzer.sum_nox_w_stations())
print("Average NOx reading considering maximum error:", nox_analyzer.average_nox_reading())