from typing import Dict, Any
import torch

from .. import Sensor


class TorchSensor(Sensor):
    def __init__(self, *pres, output=None, edges=None, label=False):
        super().__init__()
        if not edges:
            edges = []
        self.pres = pres
        self.output = output
        self.context_helper = None
        self.inputs = []
        self.edges = edges
        self.label = label
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if self.fullname not in data_item:
                self.update_pre_context(data_item)
        except:
            print('Error during updating pre data item with sensor {}'.format(self.fullname))
            raise
        self.context_helper = data_item
        try:
            data_item = self.update_context(data_item)
        except:
            print('Error during updating data item with sensor {}'.format(self.fullname))
            raise

        if self.output:
            return data_item[self.sup.sup[self.output].fullname]

        try:
            return data_item[self.fullname]
        except KeyError:
            return data_item[self.sup.sup['raw'].fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            val = self.forward()
            
        if val is not None:
            data_item[self.fullname] = val
            if not self.label:
                data_item[self.sup.fullname] = val  # override state under property name
        else:
            data_item[self.fullname] = None
            if not self.label:
                data_item[self.sup.fullname] = None
            
        if self.output:
            data_item[self.fullname] = self.fetch_value(self.output)
            data_item[self.sup.fullname] = self.fetch_value(self.output)
            
        return data_item

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for _, sensor in edge.find(Sensor):
                sensor(data_item=data_item)
        for pre in self.pres:
            if self.sup is None:
                raise ValueError('{} must be used with with property assignment.'.format(type(self)))
            for _, sensor in self.sup.sup[pre].find(Sensor):
                sensor(data_item=data_item)

    def fetch_value(self, pre, selector=None):
        if selector:
            try:
                return self.context_helper[list(self.sup.sup[pre].find(selector))[0][1].fullname]
            except:
                print("The key you are trying to access to with a selector doesn't exist")
                raise
        else:
            return self.context_helper[self.sup.sup[pre].fullname]

    def define_inputs(self):
        self.inputs = []
        for pre in self.pres:
            self.inputs.append(self.fetch_value(pre))

    def forward(self,) -> Any:
        raise NotImplementedError


class ConstantSensor(TorchSensor):
    def __init__(self, *pres, output=None, edge=None):
        super().__init__(*pres, output=output, edges=edge)

    def forward(self,) -> Any:
        return self.context_helper[self.sup.fullname]


class ReaderSensor(TorchSensor):
    def __init__(self, *pres, keyword, label=False):
        super().__init__(*pres, label=label)
        self.data = None
        self.keyword = keyword

    def fill_data(self, data):
        self.data = data[self.keyword]

    def forward(
        self,
    ) -> Any:
        if self.data is not None:
            try:
                if self.label:
                    return torch.tensor(self.data, device=self.device)
                else:
                    return self.data
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise Exception('not valid')


class NominalSensor(TorchSensor):
    def __init__(self, *pres, vocab=None, edges=None):
        super().__init__(*pres, edges=edges)
        self.vocab = vocab

    def complete_vocab(self):
        if not self.vocab:
            self.vocab = []
        value = self.forward()
        if value not in self.vocab:
            self.vocab.append(value)

    def one_hot_encoder(self, value):
        if not isinstance(value, list):
            output = torch.zeros([1, len(self.vocab)], device=self.device)
            output[0][self.vocab.index(value)] = 1
        else:
            if len(value):
                output = torch.zeros([len(value), 1, len(self.vocab)], device=self.device)
                for _it in range(len(value)):
                    output[_it][0][self.vocab.index(value[_it])] = 1
            else:
                output = torch.zeros([1, 1, len(self.vocab)], device=self.device)
        return output

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            val = self.forward()
            val = self.one_hot_encoder(val)
        if val is not None:
            data_item[self.fullname] = val
            data_item[self.sup.fullname] = val  # override state under property name
        else:
            data_item[self.fullname] = None
            data_item[self.sup.fullname] = None
        if self.output:
            data_item[self.fullname] = self.fetch_value(self.output)
            data_item[self.sup.fullname] = self.fetch_value(self.output)
        return data_item


class TorchEdgeSensor(TorchSensor):
    def __init__(self, *pres, mode="forward", keyword="default", edges=None):
        super().__init__(*pres, edges=edges)
        self.mode = mode
        self.created = 0
        self.keyword = keyword
        self.edge = None
        self.src = None
        self.dst = None
        self.edges = edges
        if mode != "forward" and mode != "backward" and mode != "selection":
            print("the mode passed to the edge sensor is not right")
            raise Exception('not valid')

    def get_initialized(self):
        self.edge = self.sup.sup
        if self.mode == "forward":
            self.src = self.edge.src
            self.dst = self.edge.dst
        elif self.mode == "backward" or self.mode == "selection":
            self.src = self.edge.dst
            self.dst = self.edge.src
        else:
            print("the mode passed to the edge sensor is not right")
            raise Exception('not valid')

    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.get_initialized()
        if not self.created:
            self.dst[self.keyword] = ConstantSensor()
            self.created = 1
        try:
            if self.fullname not in data_item:
                self.update_pre_context(data_item)
        except:
            print('Error during updating pre data_item with sensor {}'.format(self.fullname))
            raise
        self.context_helper = data_item
        try:
            data_item = self.update_context(data_item)
        except:
            print('Error during updating data_item with sensor {}'.format(self.fullname))
            raise
        return data_item[self.dst[self.keyword].fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:

        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            val = self.forward()
        if val is not None:
            data_item[self.fullname] = val
            data_item[self.dst[self.keyword].fullname] = val # override state under property name
        else:
            print("val is none")
        return data_item

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        if self.edges:
            for edge in self.edges:
                for _, sensor in edge.find(Sensor):
                    sensor(data_item)
        for pre in self.pres:
            if pre not in self.src:
                continue
            for _, sensor in self.src[pre].find(Sensor):
                sensor(data_item)
        if self.output:
            return data_item[self.output.fullname]

    def fetch_value(self, pre, selector=None):
        if selector:
            try:
                return self.context_helper[list(self.src[pre].find(selector))[0][1].fullname]
            except:
                print("The key you are trying to access to with a selector doesn't exist")
                raise
        else:
            return self.context_helper[self.src[pre].fullname]


class TorchEdgeReaderSensor(TorchEdgeSensor):
    def __init__(self, *pres, keyword, mode="forward"):
        super().__init__(*pres, mode=mode)
        self.data = None
        self.keyword = keyword

    def fill_data(self, data):
        self.data = data

    def forward(
            self,
    ) -> Any:
        if self.data:
            try:
                return self.data[self.keyword]
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise Exception('not valid')


class AggregationSensor(TorchSensor):
    def __init__(self, *pres, edges, map_key, deafault_dim = 480):
        super().__init__(*pres, edges=edges)
        self.edge_node = self.edges[0].sup
        self.map_key = map_key
        self.map_value = None
        self.data = None
        self.default_dim = deafault_dim
        if self.edges[0].name == "backward":
            self.src = self.edges[0].sup.dst
            self.dst = self.edges[0].sup.src
        else:
            print("the mode should always be passed as backward to the edge used in aggregator sensor")
            raise Exception('not valid')

    def get_map_value(self, ):
        self.map_value = self.context_helper[self.src[self.map_key].fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:

        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            self.get_map_value()
            self.get_data()
            val = self.forward()
        if val is not None:
            data_item[self.fullname] = val
            data_item[self.sup.fullname] = val # override state under property name
        return data_item

    def get_data(self):
        result = []
        for item in self.inputs[0]:
            result.append(self.map_value[item[0]:item[1]+1])
        self.data = result


class MaxAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        for item in self.data:
            results.append(torch.max(item, dim=0)[0])
        return torch.stack(results)


class MinAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        for item in self.data:
            results.append(torch.min(item, dim=0)[0])
        return torch.stack(results)


class MeanAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        if len(self.data):
            for item in self.data:
                results.append(torch.mean(item, dim=0))
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class ConcatAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        for item in self.data:
            results.append(torch.cat([x for x in item], dim=-1))
        return torch.stack(results)


class LastAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        if len(self.data):
            for item in self.data:
                results.append(item[-1])
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class FirstAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        if len(self.data):
            for item in self.data:
                results.append(item[0])
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class SelectionEdgeSensor(TorchEdgeSensor):
    def __init__(self, *pres, mode="selection"):
        super().__init__(*pres, mode=mode)
        self.selection_helper = None

    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.get_initialized()
        try:
            self.update_pre_context(data_item)
        except:
            print('Error during updating pre data item with sensor {}'.format(self.fullname))
            raise
        self.context_helper = data_item
        try:
            data_item = self.update_context(data_item)
        except:
            print('Error during updating data item with sensor {}'.format(self.fullname))
            raise
        return data_item[self.src[self.dst].fullname]

    def get_selection_helper(self):
        self.selection_helper = self.context_helper[self.src[self.dst].fullname]

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        for _, sensor in self.src[self.dst].find(Sensor):
            sensor(data_item)

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:

        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            self.get_selection_helper()
            val = self.forward()
        if val is not None:
            data_item[self.fullname] = val
            data_item[self.sup.fullname] = val  # override state under property name
        return data_item


class ProbabilitySelectionEdgeSensor(SelectionEdgeSensor):
    def forward(self,) -> Any:
        return self.selection_helper


class ThresholdSelectionEdgeSensor(SelectionEdgeSensor):
    def __init__(self, *pres, threshold=0.5):
        # FIXME: @hfaghihi, do you mean to call super class of `SelectionEdgeSensor`, so here we skip the constructor of `SelectionEdgeSensor`?
        super(SelectionEdgeSensor).__init__(*pres)
        self.threshold = threshold

    def forward(self,) -> Any:
        return torch.tensor([x for x in self.selection_helper if x >= self.threshold], device=self.device)


class ConcatSensor(TorchSensor):
    def forward(self,) -> Any:
        return torch.cat(self.inputs, dim=-1)


class ListConcator(TorchSensor):
    def forward(self,) -> Any:
        for it in range(len(self.inputs)):
            if isinstance(self.inputs[it], list):
                self.inputs[it] = torch.stack(self.inputs[it])
        return torch.cat(self.inputs, dim=-1)
