from datapizza.core.clients import Client
from datapizza.core.models import PipelineComponent
from datapizza.type import Node


class Metatagger(PipelineComponent):
    """
    A meta tagger that can tag a node.
    """

    def __init__(self, client: Client):
        self.client = client

    def tag(self, node: Node):
        return self.client.tag(node)

    def _run(self, node: Node):
        return self.tag(node)

    async def _a_run(self, node: Node):
        return await self.a_tag(node)
