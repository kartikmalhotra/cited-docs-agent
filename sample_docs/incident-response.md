# Northwind Labs Incident Response Runbook

This is a fictional sample runbook for testing cited-docs-agent.

## Severity levels

SEV1 means the product is down or customer data is at risk for many customers. SEV2 means a major feature is broken for many customers, or the product is down for one large customer. SEV3 means a minor feature is broken or slow, with a workaround available.

## Paging and response times

For a SEV1, the on-call engineer must acknowledge the page within 5 minutes and join the incident call within 15 minutes. For a SEV2, the page must be acknowledged within 15 minutes. SEV3 issues are handled during working hours through the normal ticket queue.

## Roles

Every SEV1 and SEV2 incident has an incident commander, who coordinates the response and makes decisions, and a communications lead, who keeps customers and the company informed. The incident commander is the on-call engineering manager unless they hand the role over.

## Status updates

During a SEV1, the communications lead updates the public status page at least every 30 minutes, even when there is no news. During a SEV2, updates are posted at least every hour.

## After the incident

A written postmortem is due within 5 working days of every SEV1 and SEV2. Postmortems are blameless: they describe what happened and how to prevent it, not who is at fault. Each action item from a postmortem gets an owner and a due date.
