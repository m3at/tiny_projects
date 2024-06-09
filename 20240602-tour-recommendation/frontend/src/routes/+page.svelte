<script lang="ts">
    import Autocomplete from "@smui-extra/autocomplete";
    import { Text } from "@smui/list";
    import CircularProgress from "@smui/circular-progress";
    import Button, { Label } from "@smui/button";
    import { Icon } from "@smui/icon-button";
    import { writable } from "svelte/store";
    import { onMount } from "svelte";
    import Loading from "./Loading.svelte";
    import { get } from "svelte/store";
    import Item from './Item.svelte';

    // TODO: slider for sporty <-> relaxed , as an example of customization

    // location.protocol}//${location.hostname
    let location = { protocol: "http", hostname: "127.0.0.1" };
    onMount(() => (location = window.location));
    const api_port = 8000;

    let counter = 0;

    async function fetchAutocomplete(input: string) {
        if (input === "") {
            return [];
        }

        // Pretend to have some sort of canceling mechanism.
        const myCounter = ++counter;

        // Pretend to be loading something...
        // sortof a debounce
        await new Promise((resolve) => setTimeout(resolve, 700));

        // This means the function was called again, so we should cancel.
        if (myCounter !== counter) {
            // `return false` (or, more accurately, resolving the Promise object to
            // `false`) is how you tell Autocomplete to cancel this search. It won't
            // replace the results of any subsequent search that has already finished.
            return false;
        }

        const url_api = `${location.protocol}//${location.hostname}:${api_port}/autocomplete`;

        console.log(`Autocompleting for: ${input}`);
        const response = await fetch(url_api, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            //body: JSON.stringify(data)
            body: JSON.stringify({ query: input }),
        });
        //response.headers.append('Access-Control-Allow-Origin', url_api);
        //const result = await response.text();
        const result = await response.json();
        console.log(result);
        return result;

        // Return a list of matches.
        //return fruits.filter((item) =>
        //  item.toLowerCase().includes(input.toLowerCase()),
        //);
    }

    // search
    //let search = "";
    //let products = [];
    let timeout: number | null = null;
    let searching = false;

    function handle_search() {
        console.log("in handle_search");
        searching = true;
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(fetchSearch, 300);
    }

    const placeholder_image_url = "https://placehold.co/320x180?text=16x9";

    //const items = writable([]);
    const items = writable([
        {
            id: "1",
            title: "Placeholder",
            state: 0,
            image_url: placeholder_image_url,
            location: "Placeholder",
            highlights: "Placeholder",
        },
        {
            id: "2",
            title: "Placeholder",
            state: 0,
            image_url: placeholder_image_url,
            location: "Placeholder",
            highlights: "Placeholder",
        },
        {
            id: "3",
            title: "Placeholder",
            state: 0,
            image_url: placeholder_image_url,
            location: "Placeholder",
            highlights: "Placeholder",
        },
    ]);

    function onKeyDown(e: KeyboardEvent) {
        switch (e.code) {
            case "Enter":
                console.log("key Enter pressed");
                handle_search();

                // cancel the autocomplete
                counter++;

                break;
        }
    }

    //function anyNonZeroState() {
    //    const currentItems = get(items);
    //    return currentItems.some(item => item.state !== 0);
    //}

    //$: nonZeroState = $items.some(item => item.state !== 0);
    // Create a derived state to check if any item state is non-zero
    //$: nonZeroState = get(items).some(item => item.state !== 0);
    let nonZeroState = false;
    //let nonZeroState = true;

    // Subscribe to items store to update nonZeroState
    //items.subscribe(value => {
    //    nonZeroState = value.some(item => item.state === 0);
    //});
    //items.subscribe((value) => {
    //    console.log(`ok >>> ${value}`);
    //    nonZeroState = value.some(item => item.state !== 0);
    //});
    //
    //items.set([{
    //    id: "1",
    //    title: "???",
    //    state: 0,
    //    image_url: placeholder_image_url,
    //    location: "???",
    //    highlights: "???",
    //}]);

    // Function to update items
    const updateItems = (newItems: any) => {
        const initializedItems = newItems.map((item) => ({
            ...item,
            state: 0,
        }));
        //return initializedItems;
        items.set(initializedItems);
    };

    const getItems = (newItems: any) => {
        const initializedItems = newItems.map((item) => ({
            id: item.id,
            state: item.state,
        }));
        return initializedItems;
    };

    // search button

    let search_field_content = "";
    let searching_items = false;

    function reset() {
        //products = [];
        searching_items = false;
    }

    async function fetchSearch() {
        searching_items = true;
        if (!search_field_content) {
            reset();
            return;
        }

        const url_api = `${location.protocol}//${location.hostname}:${api_port}/search`;

        //const ccc = items.get();
        const extractedItems = getItems($items);
        //var extractedItems;
        //if ($items?.length == 0) {
        //    extractedItems = [{}];
        //} else {
        //    extractedItems = $items.map((item) => ({
        //        id: item.id,
        //        state: item.state,
        //    }));
        //}
        console.log(extractedItems);
        //console.log(items);

        const response = await fetch(url_api, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            //body: JSON.stringify({"query": search_field_content, "items": $items}),
            //body: JSON.stringify({"query": search_field_content, "items": []}),
            body: JSON.stringify({
                query: search_field_content,
                items: extractedItems,
            }),
        });
        const result = await response.json();
        //console.log(result);

        // sleep a bit to simulate slow search
        //await new Promise((resolve) => setTimeout(resolve, 500));

        searching_items = false;

        updateItems(result);
        //items.set(updateItems(result));
        //items.set(result);
    }
</script>

<link
    href="https://fonts.googleapis.com/icon?family=Material+Icons"
    rel="stylesheet"
/>

<div>
    <div class="box">
        <!-- on:keydown={onKeyDown} -->
        <!-- on:input={fetchSearch} -->
        <!-- search={fetchAutocomplete} -->
        <Autocomplete
            combobox
            search={fetchAutocomplete}
            showMenuWithNoInput={false}
            label="Explore amazing tours"
            id="search"
            bind:value={search_field_content}
            on:keydown={onKeyDown}
            style="flex: 4 1 auto;"
            textfield$style="width: 100%;"
        >
            <Text
                slot="loading"
                style="display: flex; width: 100%; justify-content: center; align-items: center;"
            >
                <CircularProgress
                    style="height: 24px; width: 24px;"
                    indeterminate
                />
            </Text>
        </Autocomplete>

        <Button on:click={fetchSearch} variant="raised" style="flex: 1 2 auto">
            <Icon class="material-icons">search</Icon>
            <Label>
                {#if nonZeroState}
                    <p>Recommend</p>
                {:else}
                    <p>Search</p>
                {/if}
            </Label>
        </Button>
    </div>
</div>

<br />
<br />

{#if searching_items}
    <Loading />
{/if}

<div class="grid">
  {#each $items as item}
    <Item {item} />
  {/each}
</div>

<!--
<Grid {items} />
-->

<!--
<button on:click={fetchData}>Fetch</button>
-->

<style>
    .box {
        display: flex;
        flex-flow: row wrap;
        align-items: center;
    }
    .grid {
        display: flex;
        gap: 1rem;
        flex-flow: row wrap;
        align-items: center;
    }
</style>
