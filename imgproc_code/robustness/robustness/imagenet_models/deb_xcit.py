'''
Code from: Debenedetti, Edoardo, Vikash Sehwag, and Prateek Mittal. "A light recipe to train robust vision transformers." In 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 225-253. IEEE, 2023.
'''

from timm.models import cait, xcit, build_model_with_cfg, register_model

default_cfgs = {
    'cait_s12_224': cait._cfg(input_size=(3, 224, 224)),
    'xcit_medium_12_p16_224': xcit._cfg(),
    'xcit_large_12_p16_224': xcit._cfg(),
    'xcit_large_12_h8_p16_224': xcit._cfg(),
    'xcit_small_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
    'xcit_medium_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
    'xcit_large_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
}

def adapt_model_patches(model: xcit.Xcit, new_patch_size: int):
    to_divide = model.patch_embed.patch_size / new_patch_size
    assert int(to_divide) == to_divide, "The new patch size should divide the original patch size"
    to_divide = int(to_divide)
    assert to_divide % 2 == 0, "The ratio between the original patch size and the new patch size should be divisible by 2"
    for conv_index in range(0, to_divide, 2):
        model.patch_embed.proj[conv_index][0].stride = (1, 1)
    model.patch_embed.patch_size = new_patch_size
    return model


@register_model
def cait_s12_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=8, init_values=1.0, **kwargs)
    return build_model_with_cfg(cait.Cait,
                                'cait_s12_224',
                                pretrained,
                                pretrained_filter_fn=cait.checkpoint_filter_fn,
                                **model_kwargs)


@register_model
def xcit_medium_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_medium_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_large_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=16,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_large_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_large_12_h8_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_large_12_h8_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_small_12_p8_32(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,  # 16 because the pre-trained model has 16
        embed_dim=384,
        depth=12,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p4_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.Xcit)
    model = adapt_model_patches(model, 8)
    return model


@register_model
def xcit_small_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,  # 16 because the pre-trained model has 16
        embed_dim=384,
        depth=12,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p4_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.Xcit)
    model = adapt_model_patches(model, 4)
    return model


@register_model
def xcit_medium_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_medium_12_p4_32', pretrained=pretrained, **model_kwargs)
    # TODO: make this a function
    assert isinstance(model, xcit.Xcit)
    model = adapt_model_patches(model, 4)
    return model


@register_model
def xcit_large_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=16,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_large_12_p16_224', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.Xcit)
    model = adapt_model_patches(model, 4)
    return model


@register_model
def xcit_small_12_p2_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p2_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.Xcit)
    model = adapt_model_patches(model, 2)
    return model