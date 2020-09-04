from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, TriggerPrefilledSensor
from typing import Any


class MultiLevelReaderSensor(ConstantSensor):
    def __init__(self, *pres, keyword=None, edges=None, label=False, device='auto'):
        super().__init__(*pres, data=None, edges=edges, label=label, device=device)
        self.keyword = keyword

    def fill_data(self, data_item):
        try:
            if isinstance(self.keyword, tuple):
                self.data = (self.fetch_key(data_item, keyword) for keyword in self.keyword)
            else:
                self.data = self.fetch_key(data_item, self.keyword)
        except KeyError as e:
            raise KeyError("The key you requested from the reader doesn't exist: %s" % str(e))

    def fetch_key(self, data_item, key):
        data = []
        if "." in key:
            keys = key.split(".")
            items = data_item
            loop = 0
            direct_loop = True
            for key in keys:
                if key == "*":
                    loop += 1
                    if loop == 1:
                        keys = items.keys()
                        items = [items[key] for key in keys]
                    if loop > 1:
                        keys = [item.keys() for item in items]
                        new_items = []
                        for index, item in enumerate(items):
                            for index1, key in enumerate(keys[index]):
                                new_items.append(item[key])
                        items = new_items
                else:
                    if loop == 0:
                        items = items[key]
                    if loop > 0:
                        items = [it[key] for it in items]
            data = items
        else:
            data = data_item[key]

        return data
        
        
    def forward(self, *_) -> Any:
        if isinstance(self.keyword, tuple) and isinstance(self.data, tuple):
            return (super().forward(data) for data in self.data)
        else:
            return super().forward(self.data)
