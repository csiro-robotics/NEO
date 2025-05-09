from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from neo.data.neo_dataparser import NeoDataParserConfig

neo_dataparser = DataParserSpecification(
    config=NeoDataParserConfig(),
    description="Neo dataparser"
)