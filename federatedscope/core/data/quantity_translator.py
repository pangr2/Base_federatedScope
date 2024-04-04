from federatedscope.core.data.base_translator import BaseDataTranslator
from federatedscope.core.data.base_data import ClientData
from torch.utils.data import Dataset, Subset
SAVE = True
class QuantityDataTranslator(BaseDataTranslator):
    """
    ``FairnessDataTranslator`` convert datadict to ``StandaloneDataDict``. \
    Compared to ``core.data.base_translator.BaseDataTranslator``, it do not \
    perform FL split.
    """
    def split(self, dataset):
        """
        Perform ML split

        Returns:
            dict of ``ClientData`` with client_idx as key to build \
            ``StandaloneDataDict``
        """
        # if not isinstance(dataset, dict):
        #     raise TypeError(f'Not support data type {type(dataset)}')
        datadict = {}
        data_list = self.splitter(dataset)
        dataset_tmp = dict()
        for index, subdataset in enumerate(data_list):
            dataset_tmp[int(index+1)] = subdataset

        dataset = dataset_tmp

        for client_id in dataset.keys():
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg

            if isinstance(dataset[client_id], dict):
                datadict[client_id] = ClientData(client_cfg,
                                                 **dataset[client_id])
            else:
                # Do not have train/val/test
                train, val, test = self.split_train_val_test(dataset[client_id], client_cfg)
                tmp_dict = dict(train=train, val=val, test=test)
                # Only for graph-level task, get number of graph labels
                if client_cfg.model.task.startswith('graph') and \
                        client_cfg.model.out_channels == 0:
                    s = set()
                    for g in dataset[client_id]:
                        s.add(g.y.item())
                    tmp_dict['num_label'] = len(s)
                if SAVE:
                    save_dict = {"train": [], "val": [],
                                 "test": []}
                    for i in range(len(tmp_dict['train'])):
                        save_dict["train"].append({'input_ids': tmp_dict['train'][i]['input_ids'].tolist(),
                                                  'labels': tmp_dict['train'][i]['labels'].tolist(),
                                                  'categories': tmp_dict['train'][i]['categories'].tolist()})

                    for i in range(len(tmp_dict['val'])):
                        save_dict["val"].append({'input_ids': tmp_dict['val'][i]['input_ids'].tolist(),
                                                  'labels': tmp_dict['val'][i]['labels'].tolist(),
                                                  'categories': tmp_dict['val'][i]['categories'].tolist()})

                    for i in range(len(tmp_dict['test'])):
                        save_dict["test"].append({'input_ids': tmp_dict['test'][i]['input_ids'].tolist(),
                                                  'labels': tmp_dict['test'][i]['labels'].tolist(),
                                                  'categories': tmp_dict['test'][i]['categories'].tolist()})

                    import json

                    with open(f"{self.global_cfg.federate.save_to[:-5]}_dataset_client_{client_id}.json", "w") as outfile:
                        json.dump(save_dict, outfile)

                datadict[client_id] = ClientData(client_cfg, **tmp_dict)
        return datadict
