<script lang="ts">
  import { fade } from 'svelte/transition';
  import Icon from '$lib/components/elements/icon.svelte';
  import { mdiInformationOutline } from '@mdi/js';
  import { resolveRoute } from '$app/paths';
  import { page } from '$app/stores';
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    routeId: string;
    icon: string;
    flippedLogo?: boolean;
    isSelected?: boolean;
    preloadData?: boolean;
    moreInformation?: Snippet;
  }

  let {
    title,
    routeId,
    icon,
    flippedLogo = false,
    isSelected = $bindable(false),
    preloadData = true,
    moreInformation,
  }: Props = $props();

  let showMoreInformation = $state(false);
  let routePath = $derived(resolveRoute(routeId, {}));

  $effect(() => {
    isSelected = ($page.route.id?.match(/^\/(admin|\(user\))\/[^/]*/) || [])[0] === routeId;
  });
</script>

<a
  href={routePath}
  data-sveltekit-preload-data={preloadData ? 'hover' : 'off'}
  draggable="false"
  aria-current={isSelected ? 'page' : undefined}
  class="flex w-full place-items-center justify-between gap-4 rounded-r-full py-3 transition-[padding] delay-100 duration-100 hover:cursor-pointer hover:bg-immich-gray hover:text-immich-primary dark:text-immich-dark-fg dark:hover:bg-immich-dark-gray dark:hover:text-immich-dark-primary
    {isSelected
    ? 'bg-immich-primary/10 text-immich-primary hover:bg-immich-primary/10 dark:bg-immich-dark-primary/10 dark:text-immich-dark-primary'
    : ''}
		pl-5 group-hover:sm:px-5 md:px-5
  "
>
  <div class="flex w-full place-items-center gap-4 overflow-hidden truncate">
    <Icon path={icon} size="1.5em" class="shrink-0" flipped={flippedLogo} ariaHidden />
    <span class="text-sm font-medium">{title}</span>
  </div>

  <div
    class="h-0 overflow-hidden transition-[height] delay-1000 duration-100 sm:group-hover:h-auto group-hover:sm:overflow-visible md:h-auto md:overflow-visible"
  >
    {#if moreInformation}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="relative flex cursor-default select-none justify-center"
        onmouseenter={() => (showMoreInformation = true)}
        onmouseleave={() => (showMoreInformation = false)}
      >
        <div class="p-1 text-gray-600 hover:cursor-help dark:text-gray-400">
          <Icon path={mdiInformationOutline} />
        </div>

        {#if showMoreInformation}
          <div class="absolute right-6 top-0">
            <div
              class="flex place-content-center place-items-center whitespace-nowrap rounded-3xl border bg-immich-bg px-6 py-3 text-xs text-immich-fg shadow-lg dark:border-immich-dark-gray dark:bg-gray-600 dark:text-immich-dark-fg"
              class:hidden={!showMoreInformation}
              transition:fade={{ duration: 200 }}
            >
              {@render moreInformation?.()}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</a>
