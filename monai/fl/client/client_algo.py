# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from monai.fl.utils.exchange_object import ExchangeObject


class ClientAlgo:
    """
    objective: provide an abstract base class for defining algo to run on any platform.
    To define a new algo script, subclass this class and implement the
    following abstract methods:

        - self.train()
        - self.get_weights()
        - self.evaluate()

    initialize(), abort(), and finalize() can be optionally be implemented to help with lifecycle management
    of the class object.
    """

    def initialize(self, extra: Optional[dict] = None):
        """
        Call to initialize the ClientAlgo class

        Args:
            extra: optional extra information, e.g. dict of `ExtraItems.CLIENT_NAME` and/or `ExtraItems.APP_ROOT`
        """
        pass

    def finalize(self, extra: Optional[dict] = None):
        """
        Call to finalize the ClientAlgo class

        Args:
            extra: optional extra information
        """
        pass

    def abort(self, extra: Optional[dict] = None):
        """
        Call to abort the ClientAlgo training or evaluation

        Args:
            extra: optional extra information
        """

        pass

    def train(self, data: ExchangeObject, extra: Optional[dict] = None) -> None:
        """
        Train network and produce new network from train data.

        Args:
            data: ExchangeObject containing current network weights to base training on.
            extra: optional extra information

        Returns:
            None
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def get_weights(self, extra: Optional[dict] = None) -> ExchangeObject:
        """
        Get current local weights or weight differences

        Args:
            extra: optional extra information

        Returns:
            ExchangeObject: current local weights or weight differences.

        ExchangeObject example::

            ExchangeObject(
                weights = self.trainer.network.state_dict(),
                optim = None,  # could be self.optimizer.state_dict()
                weight_type = WeightType.WEIGHTS
            )

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def evaluate(self, data: ExchangeObject, extra: Optional[dict] = None) -> ExchangeObject:
        """
        Get evaluation metrics on test data.

        Args:
            data: ExchangeObject with network weights to use for evaluation
            extra: optional extra information

        Returns:
            metrics: ExchangeObject with evaluation metrics.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
