<script>
  export let item;

  import Card, {
    Content,
    Actions,
  } from '@smui/card';
  import { Icon } from '@smui/icon-button';

  import {
    PrimaryAction,
    Media,
    MediaContent,
  } from '@smui/card';

  import SegmentedButton, { Segment } from '@smui/segmented-button';
  import Wrapper from '@smui/touch-target';
  import {
    mdiMinus,
    mdiThumbUp,
    mdiThumbDown,
  } from '@mdi/js';
 
  const aligns = [
    {
      name: 'Left',
      icon: mdiThumbUp,
      special_value: 1,
      color: 'Green',
    },
    {
      name: 'Center',
      icon: mdiMinus,
      special_value: 0,
      color: 'Gray',
    },
    {
      name: 'Right',
      icon: mdiThumbDown,
      special_value: -1,
      color: 'Maroon',
    },
  ];

  const colors = [
    "Maroon",
    "Gray",
    "Green",
  ]
 
  let align = aligns[item.state + 1];
</script>

<div class="card-display">
    <div class="card-container">
        <Card class="result-card">
            <PrimaryAction>
                <Media class="card-media-16x9" aspectRatio="16x9" style="background-image: url({item.image_url})">
                    <MediaContent>
                        <div
                            style="color: #fff; position: absolute; bottom: 16px; left: 16px; text-shadow: 0px 0px 20px black, 0px 0px 10px black, 0px 0px 5px black, 0px 0px 3px black;"
                        >
                            <h2 class="mdc-typography--headline6" style="margin: 0;">
                                {item.title}
                            </h2>
                            <h3 class="mdc-typography--subtitle2" style="margin: 0;">
                                {item.location}
                            </h3>
                        </div>
                    </MediaContent>
                </Media>
                <Content class="mdc-typography--body2" style="overflow-y: auto; height: 80px;">
                    {item.highlights}
                </Content>
            </PrimaryAction>
            <Actions>
                <SegmentedButton
                    segments={aligns}
                    let:segment
                    singleSelect
                    bind:selected={align}
                    key={(segment) => segment.name}
                >
                    <Wrapper
                    color="secondary"
                    >
                        <Segment
                            {segment} touch title={segment.name}
                            on:click={() => item.state = segment.special_value}
                        >
                            <Icon tag="svg" style="width: 1em; height: auto;" viewBox="0 0 24 24">
                                <path fill={segment.special_value == item.state ? colors[item.state + 1] : "Gray"} d={segment.icon} />
                            </Icon>
                        </Segment>
                    </Wrapper>
                </SegmentedButton>
            </Actions>
        </Card>
    </div>
</div>

<style>
  * :global(.result-card) {
    flex: 1 1 auto;
    width: 380px;
  }
</style>
