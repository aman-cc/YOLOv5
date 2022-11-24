import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torchvision import transforms


class GeneralizedDataset:
    """
    Main class for Generalized Dataset.
    """
    
    def __init__(self, max_workers=2, verbose=False):
        self.max_workers = max_workers
        self.verbose = verbose

    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id) # PIL Image
        target = self.get_target(img_id) if self.train else {}
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        image = transforms.ToTensor()(image)
        return image, target   

    def __len__(self):
        return len(self.ids)

    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        since = time.time()
        # if os.path.exists(checked_id_file):
        #     info = [line.strip().split(", ") for line in open(checked_id_file)]
        #     self.ids, self.aspect_ratios = zip(*info)
        #     # if self.max_len != -1:
        #     #     self.ids = self.ids[:self.max_len]
        #     #     self.aspect_ratios = self.aspect_ratios[:self.max_len]
        #     print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        #     return
        
        print("Checking the dataset...")
        
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))
        
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
         
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(f'Ignoring image_id: {img_id, e}')
        return out
